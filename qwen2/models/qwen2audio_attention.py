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
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg
# from ..auto import AutoModel, AutoModelForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig, Qwen2AudioEncoderConfig

if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention
# from qwen2.models.qwen_audio_encoder import FlashAttentionKwargs
from qwen2.models.flash_attention_utils import _flash_attention_packed_forward, FlashAttentionKwargs


class Qwen2AudioFlashAttentionNoPad(Qwen2AudioAttention):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # print("Qwen2Audio Attention Layer without masking")

    @deprecate_kwarg("key_value_states", version="4.52")
    @deprecate_kwarg("past_key_value", version="4.52")
    @deprecate_kwarg("cache_position", version="4.52")
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            flash_kwargs: Optional[FlashAttentionKwargs] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        # print("no attention ")
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        #     print("Added mask...")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, None


class Qwen2AudioPackedFlashAttention(Qwen2AudioAttention):
    """
    Qwen2Audio flash attention module. This module inherits from `Qwen2AudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.whisper.modeling_whisper.WhisperFlashAttention2.__init__ with Whisper->Qwen2Audio
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

        # print("Using Qwen2Audio Flash attention")

    def _shape(self, tensor: torch.Tensor, total_batch_size: int):
        return tensor.view(total_batch_size, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reshape(self, tensor: torch.Tensor, total: int):
        return tensor.view(total, self.num_heads, self.head_dim)

    @deprecate_kwarg("key_value_states", version="4.52")
    @deprecate_kwarg("past_key_value", version="4.52")
    @deprecate_kwarg("cache_position", version="4.52")
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            flash_kwargs: Optional[FlashAttentionKwargs] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Qwen2AudioFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("Qwen2AudioFlashAttention2 attention does not support output_attentions")

        total_batch_size, _ = hidden_states.size()

        # get query proj
        # query_states = torch.reshape(self.q_proj(hidden_states), (total_batch_size, self.num_heads, self.head_dim))
        # key_states = self._shape(self.k_proj(hidden_states), total_batch_size)
        # value_states = self._shape(self.v_proj(hidden_states), total_batch_size)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout
        #  [batch_size, sequence_length, num_heads, head_dim]
        #  We would need to refactor the KV cache to be able to avoid many of these transpose/reshape/view.
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)

        query_states = self._reshape(self.q_proj(hidden_states), total_batch_size)
        key_states = self._reshape(self.k_proj(hidden_states), total_batch_size)
        value_states = self._reshape(self.v_proj(hidden_states), total_batch_size)
        # print(attention_mask, attention_mask.sum())
        # breakpoint()

        # causal_mask = attention_mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, : key_states.shape[1]]

        causal_mask = None

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_packed_forward(
            query_states,
            key_states,
            value_states,
            dropout=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            cu_seq_lens_q=flash_kwargs["cu_seq_lens_q"],
            cu_seq_lens_k=flash_kwargs["cu_seq_lens_k"],
            max_length_q=flash_kwargs["max_length_q"],
            max_length_k=flash_kwargs["max_length_k"],
            query_length=flash_kwargs["query_length"],

        )

        attn_output = attn_output.reshape(total_batch_size, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None
