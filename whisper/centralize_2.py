from memory_efficient_whisper import create_whisper_model
from bnn_lora import BayesianLinear, BayesianLoraConfig
import torch
from torch import nn
import os
import signal
import argparse
import copy
from peft import PeftModel



def centralize_and_save(model_path, lora_paths, save_path, custom_lora, save_as_lora=False):
    """
    Averages the weights of a list of PyTorch models and returns a new model with the centralized weights.

    Args:
        models (list): List of PyTorch models with identical architectures.
        lora_paths (list): List of paths to the LoRA adapters to be averaged.
        save_path (str): Path where the averaged model will be saved.
        save_as_lora (bool): If True, save as a LoRA adapter; otherwise, save the full model.

    Returns:
        torch.nn.Module: A new model with the averaged weights.
    """
    if not lora_paths:
        raise ValueError("The list of models cannot be empty.")

    device = "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


    assert len(lora_paths) > 1
    # models_state_dict = []
    models = list()

    base_model = create_whisper_model(model_path, torch_dtype,
                                      attn_implementation="sdpa",  # "flash_attention_2",
                                      low_cpu_mem_usage=True,
                                      device_map={"": device})

    main_model = base_model

    lora_path = lora_paths[0]
    if custom_lora:
        lora_config = BayesianLoraConfig.from_pretrained(lora_path)
        lora_config._register_custom_module({nn.Linear: BayesianLinear})
        main_model = PeftModel.from_pretrained(main_model, model_id=lora_path, config=lora_config)
    else:
        main_model = PeftModel.from_pretrained(main_model, lora_path)
    main_model.merge_and_unload()

    for idx, _lora_path in enumerate(lora_paths[1:]):

        sub_model = create_whisper_model(model_path, torch_dtype,
                                          attn_implementation="sdpa",  # "flash_attention_2",
                                          low_cpu_mem_usage=True,
                                          device_map={"": device})

        print(_lora_path)
        lora_weights_path = _lora_path

        # 2. Load the LoRA adapter weights onto the base model
        if custom_lora:
            lora_config = BayesianLoraConfig.from_pretrained(lora_weights_path)
            lora_config._register_custom_module({nn.Linear: BayesianLinear})
            sub_model = PeftModel.from_pretrained(main_model, model_id=lora_weights_path, config=lora_config)
        else:
            sub_model = PeftModel.from_pretrained(sub_model, lora_weights_path)
        sub_model.merge_and_unload()

        # 3. Merge the LoRA weights into the base model's weights and unload the adapter
        # models_state_dict.append(model.state_dict())
        # if idx == 0: centralized_model = copy.deepcopy(model)

        for (main_param, param) in zip(main_model.parameters(), sub_model.parameters()):

            main_param.data.add_(param.data)

    n_models = len(lora_paths)
    for main_param in main_model.parameters():
        main_param.data.div_(n_models)

    if save_as_lora:
        # print(f"Saving centralized LoRA adapter to {save_path}")
        # centralized_model.save_pretrained(save_path)
        # return centralized_model
        raise NotImplementedError
    else:
        print(f"Saving fully merged model to {save_path}")
        base_model.save_pretrained(save_path)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        original_model_path = model_path

        # Load from the original model's path
        feature_extractor = AutoFeatureExtractor.from_pretrained(original_model_path)
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)

        # Save alongside the model
        feature_extractor.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return base_model


if __name__ == "__main__":

    def handle_sigint(sig, frame):
        print("\nReceived Ctrl+C, terminating all processes...")
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_sigint)  # Handle Ctrl+C globally

    parser = argparse.ArgumentParser(description='centralize_models.py')

    parser.add_argument('-model_path', required=True, default="", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('-lora_paths', required=False, default=[], type=str,
                        nargs="+", help="Paths to the model checkpoints")
    parser.add_argument('-save_path', required=True, default="", type=str,
                        help="Path where the new model will be saved")
    parser.add_argument('-custom_lora', action='store_true', 
                        help="Use spec augmentation")
    parser.add_argument('-save_as_lora', action='store_true', 
                        help="Use spec augmentation")

    args = parser.parse_args()

    centralize_and_save(args.model_path, args.lora_paths, args.save_path, args.custom_lora, args.save_as_lora)


	
