from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft import LoraConfig
from peft.utils.other import transpose
import math

class BayesianLoraConfig(LoraConfig):
    def __init__(
        self,
        bayesian_posterior: str = None,  # e.g. "diagonal_gaussian"
        prior_std: float = 0.01,
        custom_module_class_name: str = "BayesianLinear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bayesian_posterior = bayesian_posterior
        self.prior_std = prior_std
        self.custom_module_class_name = custom_module_class_name

class BayesianRankParam(nn.Module):
    """
    Represents a diagonal Gaussian for a 2D parameter shape (rows, cols).
    We store mu, log_sigma, and sample them each forward call.
    """
    def __init__(self, rows, cols, prior_std=0.01, init_mu=0.0, init_log_sigma=-5.0):
        super().__init__()
        self.rows = cols
        self.cols = rows
        self.prior_std = prior_std

        # The means and log-std
        self.mu = nn.Parameter(torch.zeros(self.rows, self.cols))
        init_log_sigma = math.log(self.prior_std)
        self.log_sigma = nn.Parameter(torch.full((self.rows, self.cols), init_log_sigma))

        # You might add custom inits here if desired

    @property
    def weight(self) -> torch.Tensor:
        """
        Whenever someone accesses x.weight, we return the 'merged' parameter
        from multiple samples. By default, let's do 5 samples.
        """
        return self.sample_and_merge(number_of_samples=5)

    def sample_and_merge(self, number_of_samples: int = 5) -> torch.Tensor:
        """
        Draw multiple samples from the posterior, average them, and return
        a single "merged" weight matrix. This can be used to produce a final
        single update if you don't want per-inference sampling.
    
        Args:
            number_of_samples (int): How many samples to draw and average.
    
        Returns:
            torch.Tensor of shape [rows, cols]: The averaged weight matrix.
        """
        # We'll accumulate samples in a sum and then divide.
        device = self.mu.device
        sum_matrix = torch.zeros_like(self.mu, device=device)
    
        sigma = torch.exp(self.log_sigma)  # pre-compute once
        for _ in range(number_of_samples):
            eps = torch.randn_like(self.mu)
            sample = self.mu + sigma * eps
            sum_matrix += sample
    
        # Compute the average
        average_matrix = sum_matrix / number_of_samples
        return average_matrix
        
            

    def forward(self, x):
        """
        Returns a [rows, cols] sample from the posterior.
        """
        eps = torch.randn_like(self.mu)
        sigma = torch.exp(self.log_sigma)
        A_B = self.mu +sigma *eps
        return F.linear(x, A_B)
        #return self.mu + sigma * eps

    def kl_loss(self):
        """
        KL( N(mu, sigma^2) || N(0, prior_std^2) ), summed over all elements.
        """
        sigma = torch.exp(self.log_sigma)
        prior_std_t = torch.tensor(self.prior_std, device=self.mu.device)

        kl = (
            (sigma**2 + self.mu**2) / (2.0 * prior_std_t**2)
            - 0.5
            + (torch.log(prior_std_t) - self.log_sigma)
         #   + self.log_sigma
         #   - torch.log(prior_std_t)
        )
        #print(f"KL: {kl} kl.sum: {kl.sum()}", flush=True)
        return kl.sum()

class BayesianLoRALayer(LoraLayer):
    """
    Extends LoraLayer to optionally use a Bayesian posterior (e.g. diagonal Gaussian)
    for the A,B factors. 
    """

    def __init__(
        self,
        base_layer,
		ephemeral_gpu_offload: bool = False, 
      #  adapter_name: str,
      #  in_features: int,
      #  out_features: int,
      #  r: int,
      #  lora_alpha: float,
      #  lora_dropout: float = 0.0,
      #  merge_weights: bool = False,
        bayesian_posterior: str = None,  # e.g. "diagonal_gaussian" or None
        prior_std: float = 0.01,         # used if bayesian_posterior is set
        **kwargs
    ):
        #print("BayesianLoRALayer"+"=="*30)
        super().__init__(base_layer, ephemeral_gpu_offload, **kwargs)
       # super().__init__(
       #     in_features=in_features,
       #     out_features=out_features,
       #     r=r,
       #     lora_alpha=lora_alpha,
       #     lora_dropout=lora_dropout,
       #     merge_weights=merge_weights,
       #     **kwargs
       # )
        self.cast_input_dtype_enabled: bool = True
        self.bayesian_posterior = bayesian_posterior
        self.prior_std = prior_std

       # if self.bayesian_posterior == "diagonal_gaussian":
       #     # Instead of single lora_A, lora_B, store means + log-stds
       #     self.mu_A = nn.Parameter(torch.zeros(in_features, r))
       #     self.log_sigma_A = nn.Parameter(torch.full((in_features, r), -5.0))
       #     self.mu_B = nn.Parameter(torch.zeros(r, out_features))
       #     self.log_sigma_B = nn.Parameter(torch.full((r, out_features), -5.0))
       # else:
       #     # Normal LoRA approach
       #     # the parent LoraLayer might have already created self.lora_A, self.lora_B
       #     # but only if we set self.r>0
       #     pass

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
        #print(adapter_name+"=="*20)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = BayesianRankParam(self.in_features, r, prior_std=self.prior_std) #nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = BayesianRankParam(r, self.out_features, prior_std=self.prior_std, init_log_sigma=-50.0) #nn.Linear(r, self.out_features, bias=lora_bias)
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
            pass
            #print("skipping: self.reset_lora_parameters(adapter_name, init_lora_weights)")
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)


  #  def baysian_init_params(self):
  #      """
  #      Optionally re-init your Bayesian parameters, if needed. 
  #      (Not strictly necessary.)
  #      """
  #      if self.bayesian_posterior == "diagonal_gaussian":
  #          nn.init.zeros_(self.mu_A)
  #          nn.init.zeros_(self.mu_B)
  #          nn.init.constant_(self.log_sigma_A, -5.0)
  #          nn.init.constant_(self.log_sigma_B, -5.0)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Overriding the LoraLayer method to produce DeltaW in a Bayesian manner.
        Called inside forward or merge/unmerge logic in LoraLayer code.
        """
    
        # 1) Figure out device/dtype logic:
        device = self.lora_B[adapter].mu.device
        dtype = self.lora_B[adapter].mu.dtype
        cast_to_fp32 = (device.type == "cpu") and (dtype in [torch.float16, torch.bfloat16])
    
        # 2) Sample and merge => these are your final (A, B) in [in_features, r], [r, out_features]
        weight_A = self.lora_A[adapter].sample_and_merge()
        weight_B = self.lora_B[adapter].sample_and_merge()
    
        # 3) If needed, cast to float32 for the matmul
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
    
        # 4) Multiply B@A, transpose if needed, scale
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
    
        # 5) If you cast to fp32 above, cast the output back to the original dtype
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
    
            # Also cast your BayesianRankParam's mu/log_sigma back to original dtype
            # so they remain consistent with the rest of the model
            self.lora_A[adapter].mu.data = self.lora_A[adapter].mu.data.to(dtype)
            self.lora_A[adapter].log_sigma.data = self.lora_A[adapter].log_sigma.data.to(dtype)
            self.lora_B[adapter].mu.data = self.lora_B[adapter].mu.data.to(dtype)
            self.lora_B[adapter].log_sigma.data = self.lora_B[adapter].log_sigma.data.to(dtype)
    
        # 6) Return the resulting delta W
        return output_tensor



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
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

                self.merged_adapters.append(active_adapter)

       # delta_weight = None
       # if self.bayesian_posterior == "diagonal_gaussian":
       #     # Option A: Use the means for merging
       #     A = self.mu_A
       #     B = self.mu_B
       #     delta_weight = (A @ B) * self.scaling

       #     # Option B: or sample multiple times, average them
       #     # n_samples = 5
       #     # sum_w = 0.0
       #     # for _ in range(n_samples):
       #     #    sum_w += self.get_delta_weight()
       #     # delta_weight = sum_w / n_samples

       # else:
       #     # Standard LoRA approach
       #     delta_weight = (self.lora_A @ self.lora_B) * self.scaling

       # # Now we add it to the base weight
       # self.weight.data += delta_weight.data
       # self.merged = True

    def unmerge(self):
        """
        Unmerge the LoRA from the base weight. 
        If merged, subtract the same delta_weight from the base weight. 
        """
        if not self.merged:
            return

        # We subtract the same delta weight we added in merge
        if self.bayesian_posterior == "diagonal_gaussian":
            A = self.mu_A
            B = self.mu_B
            delta_weight = (A @ B) * self.scaling
        else:
            delta_weight = (self.lora_A @ self.lora_B) * self.scaling

        self.weight.data -= delta_weight.data
        self.merged = False

#    def kl_loss(self) -> torch.Tensor:
#        """
#        If using Bayesian posterior, compute KL(q(A,B) || p(A,B)).
#        Otherwise, return 0.
#        """
#        if self.bayesian_posterior != "diagonal_gaussian":
#            return torch.tensor(0.0, device=self.weight.device)
#
#        # Diagonal Gaussian KL
#        sigma_A = torch.exp(self.log_sigma_A)
#        sigma_B = torch.exp(self.log_sigma_B)
#        prior_std_t = torch.tensor(self.prior_std, device=self.weight.device)
#
#        kl_A = (
#            (sigma_A**2 + self.mu_A**2) / (2.0 * prior_std_t**2)
#            - 0.5
#            + self.log_sigma_A
#            - torch.log(prior_std_t)
#        )
#        kl_B = (
#            (sigma_B**2 + self.mu_B**2) / (2.0 * prior_std_t**2)
#            - 0.5
#            + self.log_sigma_B
#            - torch.log(prior_std_t)
#        )
#        return kl_A.sum() + kl_B.sum()

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)

    def __repr__(self):
        base = super().__repr__()
        if self.bayesian_posterior == "diagonal_gaussian":
            base += f"\n  Bayesian posterior: diagonal_gaussian (prior_std={self.prior_std})"
        return base


class BayesianLinear(nn.Module, BayesianLoRALayer):
    """
    An example that merges the standard `nn.Linear` forward with `BayesianLoRALayer` logic.
    Typically used if you want to fully replace a standard linear with this Bayesian-lora variant.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
       # in_features: int,
       # out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        bias: bool = True,
        bayesian_posterior: str = None,
        prior_std: float = 0.01,
        **kwargs,
    ):
        #print("BayesianLinear"+"==="*40)
        super().__init__()
        BayesianLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
       # nn.Module.__init__(self)
     #   BayesianLoRALayer.__init__(
     #       self,
     #       in_features=in_features,
     #       out_features=out_features,
     #       r=r,
     #       lora_alpha=lora_alpha,
     #       lora_dropout=lora_dropout,
     #       merge_weights=False,
     #       bayesian_posterior=bayesian_posterior,
     #       prior_std=prior_std,
     #       **kwargs
     #   )
        # Create the base weight
        #self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        #self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        #if lora_dropout > 0.0:
        #    self.lora_dropout = nn.Dropout(p=lora_dropout)
        #else:
        #    self.lora_dropout = None

        #self.reset_parameters()
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)      
        #print(kwargs)
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
            for active_adapter in self.active_adapters:
                #print(f"ACTIVE ADAPTER: {active_adapter}")
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                #print(f"LORA A MU: {lora_A.mu}")
                x = self._cast_input_dtype(x, lora_A.mu.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)

        return result


    def chatgpt_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # If merged, do normal linear
            out = x @ self.weight.T
        else:
            # else do normal weight + get_delta_weight
            delta_w = self.get_delta_weight()  # sampling or deterministic
            effective_weight = self.weight + delta_w
            out = x @ effective_weight.T

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return (f"BayesianLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, r={self.r}, "
                f"bayesian_posterior={self.bayesian_posterior}, "
                f"bias={self.bias is not None})")
