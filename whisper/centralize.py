from memory_efficient_whisper import create_whisper_model
import torch
import os
import signal
import argparse
import copy
from peft import PeftModel

def load_base_model(rank, model_path, lora_path,
                          device_id,):
    """Loads model on specific GPU and decodes its chunk."""
    #torch.cuda.set_device(device_id)

   # device = torch.device(f"cuda:{rank}")
    #device = device if torch.cuda.is_available() else "cpu"
    device = "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = create_whisper_model(model_path, torch_dtype,
                                 attn_implementation="sdpa", #"flash_attention_2",
                                 low_cpu_mem_usage=True,
                                 device_map={"": device})
    return model


def centralize_and_save(base_model, lora_paths, save_path):
	"""
    Averages the weights of a list of PyTorch models and returns a new model with the centralized weights.

    Args:
        models (list): List of PyTorch models with identical architectures.

    Returns:
        torch.nn.Module: A new model with the averaged weights.
	"""
	if not lora_paths:
		raise ValueError("The list of models cannot be empty.")

	models_state_dict = []
	for idx, lora_path in enumerate(lora_paths):
		print(lora_path)
		lora_weights_path = lora_path
		
		# 2. Load the LoRA adapter weights onto the base model
		model = PeftModel.from_pretrained(base_model, lora_weights_path)

		# 3. Merge the LoRA weights into the base model's weights and unload the adapter
		#model.merge_and_unload()
		models_state_dict.append(model.state_dict())
		if idx == 0: centralized_model = copy.deepcopy(model)

	centralized_model_state_dict = copy.deepcopy(models_state_dict[0])

	# Iterate through the keys (weights) and average them
	for key in centralized_model_state_dict.keys():
		centralized_model_state_dict[key] = sum(d[key] for d in models_state_dict) / len(models_state_dict)

	# Load the averaged weights into the new model
	centralized_model.load_state_dict(centralized_model_state_dict)

	centralized_model.save_pretrained(save_path)
	return centralized_model


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

    args = parser.parse_args()

    base_model = load_base_model(0, args.model_path, args.lora_paths, 0,)

    centralize_and_save(base_model, args.lora_paths, args.save_path)

	
