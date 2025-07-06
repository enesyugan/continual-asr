# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2Audio model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, TypedDict

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from optimized.layer_norm import MemoryEfficientLayerNorm

from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig, Qwen2AudioEncoderConfig
# from transformers.models.qwen2_audio.modeling_qwen2_audio import QWEN2AUDIO_ATTENTION_CLASSES
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder, Qwen2AudioPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from transformers.activations import ACT2FN
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb  # noqa
else:
    index_first_axis = None
    pad_input = None

from qwen2.models.qwen2audio_attention import Qwen2AudioPackedFlashAttention
from qwen2.models.flash_attention_utils import FlashAttentionKwargs


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention

QWEN2AUDIO_ATTENTION_CLASSES_FIX = {
    "eager": Qwen2AudioAttention,
    "flash_attention_2": Qwen2AudioPackedFlashAttention,
    "sdpa": Qwen2AudioAttention
}


class MemoryOptimizedQwen2AudioEncoderLayer(Qwen2AudioEncoderLayer):
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)

        self.self_attn = QWEN2AUDIO_ATTENTION_CLASSES_FIX[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        flash_kwargs=None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        assert self._attn_implementation == "flash_attention_2" or flash_kwargs is None
        attn_args = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "layer_head_mask": layer_head_mask,
            "output_attentions": output_attentions
        }

        if self._attn_implementation == "flash_attention_2":
            attn_args["flash_kwargs"] = flash_kwargs

        hidden_states, attn_weights, _ = self.self_attn(
            # hidden_states=hidden_states,
            # attention_mask=attention_mask,
            # layer_head_mask=layer_head_mask,
            # output_attentions=output_attentions,
            **attn_args
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class MemoryOptimizedQwen2AudioEncoder(Qwen2AudioEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2AudioEncoderLayer`].

    Args:
        config: Qwen2AudioEncoderConfig
    """
    _no_split_modules = ["MemoryOptimizedQwen2AudioEncoderLayer"]

    def __init__(self, config: Qwen2AudioEncoderConfig):
        # ignore the init of Qwen2AudioEncoder
        Qwen2AudioPreTrainedModel.__init__(self, config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList(
            [MemoryOptimizedQwen2AudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self._attn_implementation = config._attn_implementation

    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            flash_kwargs=None
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Qwen2Audio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Qwen2Audio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # TODO: flash attention unpadding

        batch_size, seq_len, hidden_size = hidden_states.shape

        if self._attn_implementation == "flash_attention_2" and not self.gradient_checkpointing:

            # TODO: prepare data for flash attention
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)

            indices_k, cu_seq_lens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

            query_len = seq_len

            flattened = hidden_states.contiguous().view(-1, hidden_states.size(-1))
            packed_hidden_states = index_first_axis(flattened, indices_k)
            flash_kwargs: FlashAttentionKwargs = {
                "cu_seq_lens_q": cu_seq_lens_k,
                "cu_seq_lens_k": cu_seq_lens_k,
                "max_length_q": max_seqlen_in_batch_k,
                "max_length_k": max_seqlen_in_batch_k,
                "query_length": query_len,
                "totals": packed_hidden_states.size(0)
            }

            hidden_states = packed_hidden_states
        else:
            indices_k = None
            flash_kwargs = None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        flash_kwargs=flash_kwargs
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self._attn_implementation == "flash_attention_2" and not self.gradient_checkpointing:
            # pad the data again
            hidden_states = pad_input(hidden_states, indices_k, batch_size, seq_len)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
