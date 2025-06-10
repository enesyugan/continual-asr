from io import BytesIO
import os, sys
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor
from transformers import AutoConfig
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qwen2.qwen2model import create_qwen2audio_model

from transformers.models.qwen2_audio.modeling_qwen2_audio import QWEN2AUDIO_ATTENTION_CLASSES

from qwen2.models.qwen2audio_attention import Qwen2AudioFlashAttentionNoPad

# test command:
# python -m qwen2.test_qwen

# Inject your class into the registry
QWEN2AUDIO_ATTENTION_CLASSES["flash_attention_2"] = Qwen2AudioFlashAttentionNoPad
# QWEN2AUDIO_ATTENTION_CLASSES["eager"] = Qwen2AudioAttentionNoMask

# attention_type = "eager"
attention_type = "flash_attention_2"

config = AutoConfig.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

print(config)

ds = load_dataset("LIUM/tedlium", "release3")
ds = ds.filter(
    lambda x: x["text"].strip().lower() not in ["ignore_time_segment_in_scoring", "ignore_time_segment_in_scoring."],
    num_proc=8)
dev_set = ds["validation"]

sample = dev_set[0]

for key in sample.keys():
    print(key, ":", sample[key])

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
print(processor.tokenizer.eos_token)
print(processor.tokenizer.eos_token_id)
print(processor.tokenizer.bos_token)
print(processor.tokenizer.bos_token_id)
print(processor.tokenizer.pad_token)
print(processor.tokenizer.pad_token_id)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
audio, sr = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)

# print(audio.shape)
# breakpoint()

# for key in inputs:
#     print(key, inputs[key].size())
#     inputs[key] = inputs[key].to(device)
#
# print(inputs['input_ids'])

model = create_qwen2audio_model("Qwen/Qwen2-Audio-7B", config,
                                torch_dtype=torch_dtype,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                attn_implementation=attention_type,
                                mem_efficient=True
                                )

torch.compile(model)
model = model.to(device)


def generate_batches(dataset, batch_size):
    batch = []
    for sample in dataset:
        if sample['text'] and sample['audio'] is not None:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


with torch.inference_mode():
    # for i in range(len(dev_set)):

    for batch in generate_batches(dev_set, batch_size=1):

        # prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech in English:"
        prompts = [
            "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
            for _ in batch
        ]

        # audio = dev_set[i]['audio']['array']
        audios = [sample['audio']['array'] for sample in batch]
        refs = [sample['text'] for sample in batch]

        # print(audio.size())

        # inputs = processor(text=prompt, audio=audio, return_tensors="pt", add_special_tokens=True, sampling_rate=16000)
        # Process as a batch
        inputs = processor(
            text=prompts,
            audio=audios,
            return_tensors="pt",
            add_special_tokens=True,
            sampling_rate=16000,
            padding=True  # <- pads input_ids and audio
        )

        # move inputs to device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        generated_ids = model.generate(
            **inputs,
            max_length=1024,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=0.9,
            early_stopping=True,
            # 🔥 Remove top_k / top_p when not sampling
            # top_k=0,
            # top_p=0.01,
        )

        # Remove prompt tokens
        prompt_lens = inputs["input_ids"].ne(processor.tokenizer.pad_token_id).sum(dim=1)
        clean_outputs = [
            gen_ids[prompt_len:] for gen_ids, prompt_len in zip(generated_ids, prompt_lens)
        ]

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]

        # Decode
        responses = processor.batch_decode(
            clean_outputs,
            skip_cspecial_tokens=True,
            clean_up_tokenization_spaces=True
        )

        def strip_known_tags(text):
            return text.replace("<|en|>", "").replace("<|endoftext|>", "").strip()

        responses = [strip_known_tags(d) for d in responses]

        for hypo, ref in zip(responses, refs):
            print("Hypo:", hypo)
            print("Ref: ", ref)
            print("")
