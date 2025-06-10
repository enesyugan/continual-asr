from memory_efficient_whisper import create_whisper_model
import torch
from torch import nn
import os, sys
import signal
import argparse
import copy
from peft import PeftModel

from decode import find_weight_path
from bnn_lora import BLoBConfig, BLoB, BLoBModel
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from peft import get_peft_model_state_dict


def centralize_and_save(model_path, lora_paths, save_path,
                        custom_lora, save_as_lora=False,
                        scale_by_variance=False,
                        auto_find_checkpoint="none"):
    """
    Averages the weights of a list of PyTorch models and returns a new model with the centralized weights.
    Args:
        model_path:
        lora_paths:
        save_path:
        custom_lora:
        save_as_lora:
        auto_find_checkpoint:

    Returns:

    """
    if not lora_paths:
        raise ValueError("The list of models cannot be empty.")

    lora_paths = lora_paths.split("|")

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
    main_state_dict = None
    denominators = dict()

    lora_path = lora_paths[0]

    weight_path = str(find_weight_path(lora_path, auto_find_checkpoint))

    print("[INFO] Loading LORA weights from {}".format(weight_path))

    # if custom_lora:
    #     lora_config = BayesianLoraConfig.from_pretrained(lora_path)
    #     lora_config._register_custom_module({nn.Linear: BayesianLinear})
    #     main_model = PeftModel.from_pretrained(main_model, model_id=lora_path, config=lora_config)
    # else:
    #     main_model = PeftModel.from_pretrained(main_model, lora_path)
    # main_model.merge_and_unload()
    if custom_lora:

        LoraModel._create_and_replace = BLoBModel._create_and_replace

        lora_config = BLoBConfig.from_pretrained(weight_path)
        print(lora_config)
        lora_config._register_custom_module({nn.Linear: BLoB})
        main_model = PeftModel.from_pretrained(main_model, model_id=weight_path, config=lora_config)

        main_state_dict = get_peft_model_state_dict(main_model)
        # main_model.merge_and_unload()

        for key in main_state_dict.keys():

            if key.endswith(".mu"):

                sigma_key = key.replace(".mu", ".log_sigma")

                # store the denominators, initialized as inverse variance
                # because variance = uncertainty -> inverse variance = importance
                inv_variance = 1 / torch.exp(2 * main_state_dict[sigma_key])
                # inv_variance = torch.exp(2 * main_state_dict[sigma_key])
                denominators[key] = inv_variance
                if scale_by_variance:
                    main_state_dict[key] = main_state_dict[key] * inv_variance
    else:
        main_model = PeftModel.from_pretrained(main_model, weight_path)
        main_model.merge_and_unload()

    for idx, _lora_path in enumerate(lora_paths[1:]):

        sub_model_state_dict = dict()

        sub_model = create_whisper_model(model_path, torch_dtype,
                                         attn_implementation="sdpa",  # "flash_attention_2",
                                         low_cpu_mem_usage=True,
                                         device_map={"": device})

        _weight_path = str(find_weight_path(_lora_path, auto_find_checkpoint))
        print("[INFO] Loading LORA weights from {}".format(_weight_path))

        if custom_lora:

            LoraModel._create_and_replace = BLoBModel._create_and_replace

            lora_config = BLoBConfig.from_pretrained(_weight_path)
            print(lora_config)
            lora_config._register_custom_module({nn.Linear: BLoB})
            sub_model = PeftModel.from_pretrained(sub_model, model_id=_weight_path, config=lora_config)

            # sub_model.merge_and_unload()

        else:
            sub_model = PeftModel.from_pretrained(sub_model, _weight_path)
            # sub_model.merge_and_unload()

        sub_model_state_dict = get_peft_model_state_dict(sub_model)

        # TODO: try different ideas of linear interpolation
        # for (main_param, param) in zip(main_model.parameters(), sub_model.parameters()):
        #     main_param.data.add_(param.data)
        for key in main_state_dict.keys():

            if key.endswith(".mu"):

                sigma_key = key.replace(".mu", ".log_sigma")

                inv_variance = 1 / torch.exp(2 * sub_model_state_dict[sigma_key])
                # inv_variance = torch.exp(2 * sub_model_state_dict[sigma_key])
                denominators[key] += inv_variance

                # TODO: averaging based on
                if scale_by_variance:
                    main_state_dict[key] += inv_variance * sub_model_state_dict[key]
                else:
                    main_state_dict[key] += sub_model_state_dict[key]
            elif key.endswith(".log_sigma"):

                pass
            else:

                # averaging the weights for everything else :)
                main_state_dict[key] += sub_model_state_dict[key]

    # averaging
    n_models = len(lora_paths)
    for key in main_state_dict.keys():

        if key.endswith(".mu"):

            if scale_by_variance:
                # print(scale_by_variance)
                main_state_dict[key].div_(denominators[key])
            else:
                main_state_dict[key].div_(n_models)

        elif key.endswith(".log_sigma"):

            pass
        else:
            main_state_dict[key].div_(n_models)

    main_model.load_state_dict(main_state_dict, strict=False)
    main_model.merge_and_unload()

    main_model = main_model.base_model.model
    main_model.config.forced_decoder_ids = None

    if save_as_lora:
        # print(f"Saving centralized LoRA adapter to {save_path}")
        # centralized_model.save_pretrained(save_path)
        # return centralized_model
        raise NotImplementedError
    else:
        print(f"Saving fully merged model to {save_path}")
        print(type(main_model))
        # print(main_model.config.forced_decoder_ids)
        main_model.save_pretrained(save_path)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        original_model_path = model_path

        # Load from the original model's path
        feature_extractor = AutoFeatureExtractor.from_pretrained(original_model_path)
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)

        # Save alongside the model
        feature_extractor.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return main_model


if __name__ == "__main__":
    def handle_sigint(sig, frame):
        print("\nReceived Ctrl+C, terminating all processes...")
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_sigint)  # Handle Ctrl+C globally

    parser = argparse.ArgumentParser(description='centralize_models.py')

    parser.add_argument('-model_path', required=True, default="", type=str,
                        help="Path to the model checkpoint")
    # parser.add_argument('-lora_paths', required=False, default=[], type=str,
    #                     nargs="+", help="Paths to the model checkpoints")
    parser.add_argument('-lora_path', required=False, default="", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('-save_path', required=True, default="", type=str,
                        help="Path where the new model will be saved")
    parser.add_argument('-custom_lora', action='store_true',
                        help="Use spec augmentation")
    parser.add_argument('-save_as_lora', action='store_true',
                        help="Use spec augmentation")

    parser.add_argument('-scale_by_variance', action='store_true',
                        help="We can use ")

    parser.add_argument('-auto_find_checkpoint', default="none", type=str,
                        help="Automatically find checkpoint to easily use with huggingface training/tuning. "
                             "Options: none|best|latest")

    args = parser.parse_args()

    centralize_and_save(args.model_path, args.lora_path, args.save_path,
                        args.custom_lora, args.save_as_lora,
                        args.scale_by_variance,
                        args.auto_find_checkpoint)
