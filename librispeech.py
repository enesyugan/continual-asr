import torch
from datasets import load_dataset
# ds = load_dataset("librispeech_asr", "clean", split="validation")

# ds = load_dataset("LIUM/tedlium", "release3", split="train")

# ds = load_dataset(
#     "librispeech_asr",
#     "clean",
#     split="validation",
#     download_config={"max_retries": 5, "timeout": 60}
# )

ds = load_dataset("LIUM/tedlium", "release3")

dev_set = ds["validation"]

audio = torch.FloatTensor(dev_set[0]['audio']['array'])

print(audio.size())