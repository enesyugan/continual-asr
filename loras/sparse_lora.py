import torch

from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft import LoraConfig
from peft.utils.other import transpose
from peft.utils.other import get_pattern_key
from peft.utils import get_quantization_config
import math
import operator
import json

from loras.sparse_linear import spspmm_autograd


def kaiming_uniform_sparse(values: torch.Tensor,
                           fan_in: int,
                           fan_out: int,
                           a=0,
                           nonlinearity="leaky_relu"):
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std  # Uniform(-bound, bound)
    return values.uniform_(-bound, bound)


class SparseConfig(LoraConfig):
    def __init__(
            self,
            density=0.001,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.density = density

    def __repr__(self):
        base_repr = super().__repr__()
        custom_fields = (
            f"density={self.density!r}, "

        )
        return f"{self.__class__.__name__}({custom_fields}, {base_repr})"

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({

            "density": self.density
        })

        # Convert sets to lists for JSON compatibility
        for key, value in base_dict.items():
            if isinstance(value, set):
                base_dict[key] = list(value)

        return base_dict

    @classmethod
    def from_dict(cls, config_dict):

        print(config_dict)
        density = config_dict.pop("density", 0.01 * 0.01)
        # print("✅ BLoB received:")
        # print("prior_std =", prior_std)
        # print("init_log_sigma =", init_log_sigma)
        # print("bayesian_a_only =", bayesian_a_only)
        # breakpoint()

        return cls(
            density=density,
            **config_dict
        )

    def save_pretrained(self, save_directory):

        base_dict = self.to_dict()
        config_json = json.dumps(base_dict, indent=2)
        with open(f"{save_directory}/adapter_config.json", "w") as f:
            f.write(config_json)

    @classmethod
    def from_pretrained(cls, load_directory):
        with open(f"{load_directory}/adapter_config.json", "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class SparseLinearLayer(nn.Module):
    """
    Bayesian Low-Rank by Backprop (a version of BBB)
    Represents a diagonal Gaussian for a 2D parameter shape (rows, cols).
    We store mu, log_sigma, and sample them each forward call.
    """

    def __init__(self,
                 rows: int,
                 cols: int,
                 density=1e-4,
                 sparsity="coo",
                 dtype=torch.float32,
                 device='cpu'):
        super().__init__()
        self.rows = cols
        self.cols = rows

        # This represents the batch_ensembles-ed A matrix in Lora. B is kept deterministic.

        self.density = density
        self.sparsity = sparsity # can be either COO or CSR

        nnz = self.nnz = max(1, min(int(density * rows * cols), rows * cols))

        # The means and log-std
        # Random unique indices
        row_indices = torch.randint(0, rows, size=(nnz,), device=device)
        col_indices = torch.randint(0, cols, size=(nnz,), device=device)

        self.indices = torch.stack([row_indices, col_indices])  # [2, nnz]

        # Random values
        self.values = torch.randn(nnz, dtype=dtype, device=device, requires_grad=True)

        # Create sparse tensor
        # sparse = torch.sparse_coo_tensor(self.indices, values, size=(rows, cols), device=device)
        self.shape = (rows, cols)

        self.reset_parameters()
        # You might add custom inits here if desired

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.kaiming_uniform_(self.mu, a=math.sqrt(5), nonlinearity="linear")
        kaiming_uniform_sparse(self.values, self.shape[0], self.shape[1],
                               nonlinearity="linear")

    @property
    def weight(self) -> torch.Tensor:
        """
        Whenever someone accesses x.weight, we return the 'merged' parameter
        from multiple samples. By default, let's do 5 samples.
        """
        # return self.sample_and_merge(number_of_samples=32)
        return torch.sparse_coo_tensor(self.indices, self.values, self.shape).coalesce().to_sparse_csr()

    def forward(self, x):
        """
        Returns a [rows, cols] sample from the posterior.
        """
        # eps = torch.randn_like(self.mu)
        # sigma = torch.exp(self.log_sigma)
        # A_B = self.mu +sigma *eps
        # print("forward bayes lora")
        # return F.linear(x, self.mu)

        raise NotImplementedError

        return F.linear(x, self.weight.t())

    def print_grads(self):

        pass

class SparseLinear(Linear):
    """
    Extends LoraLayer to optionally use a Bayesian posterior (e.g. diagonal Gaussian)
    for the A,B factors.
    """

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            lora_bias: bool = False,
            **kwargs
    ):
        self.density = kwargs.pop("density", None)

        # print("✅ BLoB received:")
        # print("bayesian_posterior =", self.bayesian_posterior)
        # print("prior_std =", self.prior_std)
        # print("init_log_sigma =", self.init_log_sigma)
        # print("bayesian_a_only =", self.bayesian_a_only)
        # print("trick ", self.trick)
        # breakpoint()

        # print(r, lora_alpha)
        # print("BayesianLoRALayer"+"=="*30)
        super().__init__(base_layer, adapter_name,
                         r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, init_lora_weights=init_lora_weights,
                         use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias,
                         **kwargs)

    def update_layer(
            self,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora: bool = False,
            lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        # print(adapter_name+"=="*20)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters

        # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_A[adapter_name] = SparseLinearLayer(self.in_features, r, density=self.density)

        # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_B[adapter_name] = SparseLinearLayer(self.in_features, r, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
            # pass
            # print("skipping: self.reset_lora_parameters(adapter_name, init_lora_weights)")
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.mu = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def reset_lora_parameters(self, adapter_name, init_lora_weights):

        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].mu, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")

            # if self.bayesian_a_only:
            #     nn.init.zeros_(self.lora_B[adapter_name].weight)
            # else:
            #     # for Bayesian LoRA B, we do init as zero
            #     # however with the noise injected, the purpose of zero initialization is not intact
            nn.init.zeros_(self.lora_B[adapter_name].values)

            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        return super().merge(safe_merge=safe_merge, adapter_names=adapter_names)

        # TODO: custom merge?



    def unmerge(self):
        """
        Unmerge the LoRA from the base weight.
        If merged, subtract the same delta_weight from the base weight.
        """

        super().unmerge()

        # TODO: unmerge() batch_ensembles style?


    def __repr__(self):
        base = super().__repr__()
        base += f"\n  Sparse Linear (density={self.density})"
        return base

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    AB = spspmm_autograd(lora_A.indices, lora_A.values, lora_A.shape,
                                         lora_B.indices, lora_B.values, lora_B.shape)

                    result = result + F.linear(x, AB.t())

                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )

            result = result.to(torch_result_dtype)

        return result


# class BayesianLinear(nn.Module, BayesianLoRALayer):
#     """
#     An example that merges the standard `nn.Linear` forward with `BayesianLoRALayer` logic.
#     Typically used if you want to fully replace a standard linear with this Bayesian-lora variant.
#     """
#
#     def __init__(
#         self,
#         base_layer,
#         adapter_name: str,
#        # in_features: int,
#        # out_features: int,
#         r: int = 0,
#         lora_alpha: float = 1.0,
#         lora_dropout: float = 0.0,
#         fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         is_target_conv_1d_layer: bool = False,
#         init_lora_weights: Union[bool, str] = True,
#         use_rslora: bool = False,
#         use_dora: bool = False,
#         lora_bias: bool = False,

#         bias: bool = True,
#         bayesian_posterior: str = None,
#         prior_std: float = 0.01,
#         **kwargs,
#     ):
#         #print("BayesianLinear"+"==="*40)
#         super().__init__()
#         BayesianLoRALayer.__init__(self, base_layer, **kwargs)
#         self.fan_in_fan_out = fan_in_fan_out
#
#         #self.reset_parameters()
#         self._active_adapter = adapter_name
#         self.update_layer(
#             adapter_name,
#             r,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#             init_lora_weights=init_lora_weights,
#             use_rslora=use_rslora,
#             use_dora=use_dora,
#             lora_bias=lora_bias,
#         )
#         self.is_target_conv_1d_layer = is_target_conv_1d_layer
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
#         self._check_forward_args(x, *args, **kwargs)
#         adapter_names = kwargs.pop("adapter_names", None)
#         #print(kwargs)
#         if self.disable_adapters:
#             if self.merged:
#                 self.unmerge()
#             result = self.base_layer(x, *args, **kwargs)
#         elif adapter_names is not None:
#             result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
#         elif self.merged:
#             result = self.base_layer(x, *args, **kwargs)
#         else:
#             result = self.base_layer(x, *args, **kwargs)
#             torch_result_dtype = result.dtype
#             for active_adapter in self.active_adapters:
#                 #print(f"ACTIVE ADAPTER: {active_adapter}")
#                 if active_adapter not in self.lora_A.keys():
#                     continue
#                 lora_A = self.lora_A[active_adapter]
#                 lora_B = self.lora_B[active_adapter]
#                 dropout = self.lora_dropout[active_adapter]
#                 scaling = self.scaling[active_adapter]
#                 #print(f"LORA A MU: {lora_A.mu}")
#                 # x = self._cast_input_dtype(x, lora_A.mu.dtype)
#                 x = self._cast_input_dtype(x, lora_B.weight.dtype)
#
#                 if not self.use_dora[active_adapter]:
#                     # print("forward batch_ensembles lora ....")
#                     result = result + lora_B(lora_A(dropout(x))) * scaling
#                 else:
#                     if isinstance(dropout, nn.Identity) or not self.training:
#                         base_result = result
#                     else:
#                         x = dropout(x)
#                         base_result = None
#
#                     result = result + self.lora_magnitude_vector[active_adapter](
#                         x,
#                         lora_A=lora_A,
#                         lora_B=lora_B,
#                         scaling=scaling,
#                         base_layer=self.get_base_layer(),
#                         base_result=base_result,
#                     )
#
#             result = result.to(torch_result_dtype)
#
#         return result
#
#
#     def chatgpt_forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.merged:
#             # If merged, do normal linear
#             out = x @ self.weight.T
#         else:
#             # else do normal weight + get_delta_weight
#             delta_w = self.get_delta_weight()  # sampling or deterministic
#             effective_weight = self.weight + delta_w
#             out = x @ effective_weight.T
#
#         if self.bias is not None:
#             out = out + self.bias
#
#         return out
#
#     def __repr__(self):
#         return (f"BayesianLinear(in_features={self.in_features}, "
#                 f"out_features={self.out_features}, r={self.r}, "
#                 f"bayesian_posterior={self.bayesian_posterior}, "
#                 f"bias={self.bias is not None})")

class BLoBModel(LoraModel):

    def __init__(self, model, config, adapter_name, **kwargs):
        super().__init__(model, config, adapter_name, **kwargs)

    def _create_and_replace(
            self,
            lora_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

        # Quan changes the kwargs to add blob config
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "prior_std": lora_config.prior_std,
            "init_log_sigma": lora_config.init_log_sigma,
            "bayesian_a_only": lora_config.bayesian_a_only,
            "trick": lora_config.trick
        }
        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias

            )
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
