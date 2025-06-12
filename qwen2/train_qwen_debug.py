import os
import sys

import torch
import torchaudio
import yaml
from datasets import concatenate_datasets
from transformers import AutoConfig
from transformers import AutoProcessor
from transformers.models.qwen2_audio.modeling_qwen2_audio import QWEN2AUDIO_ATTENTION_CLASSES

# Local Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qwen2.qwen2model import create_qwen2audio_model
from qwen2.models.qwen2audio_attention import Qwen2AudioFlashAttentionNoPad

from qwen2.stm_to_dataset import get_train_dev
from qwen2.collator import DataCollatorForQwen2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# test command:
# python -m qwen2.test_qwen

# Inject your class into the registry
QWEN2AUDIO_ATTENTION_CLASSES["flash_attention_2"] = Qwen2AudioFlashAttentionNoPad

attention_type = "flash_attention_2"

config = AutoConfig.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

print(config)

with open("fisher.yaml", "r") as f:
    data_config = yaml.safe_load(f)
    train_data_dict, dev_data = get_train_dev(data_config)

train_data = concatenate_datasets(list(train_data_dict.values()))

sample = dev_data[0]
audio, sampling_rate = torchaudio.load(dev_data[0]['wav_path'])
print(audio.size())
# ds = load_dataset("LIUM/tedlium", "release3")
# ds = ds.filter(
#     lambda x: x["text"].strip().lower() not in ["ignore_time_segment_in_scoring", "ignore_time_segment_in_scoring."],
#     num_proc=8)
# dev_set = ds["validation"]

# sample = dev_set[0]
#
# for key in sample.keys():
#     print(key, ":", sample[key])

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
print(processor.tokenizer.eos_token)
print(processor.tokenizer.eos_token_id)
print(processor.tokenizer.bos_token)
print(processor.tokenizer.bos_token_id)
print(processor.tokenizer.pad_token)
print(processor.tokenizer.pad_token_id)
audio_eos_token = "<|audio_eos|>"
audio_eos_id = processor.tokenizer.convert_tokens_to_ids(audio_eos_token)
print(audio_eos_id)
# print(processor.tokenizer.additional_special_tokens)

# print(processor.tokenizer.special_tokens_map)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"

model = create_qwen2audio_model("Qwen/Qwen2-Audio-7B", config,
                                torch_dtype=torch_dtype,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                attn_implementation=attention_type,
                                mem_efficient=True
                                )

# torch.compile(model)
model = model.to(device)
from peft import LoraConfig

lora_target_modules = ["q_proj", "v_proj"]

lora_config = LoraConfig(r=32, lora_alpha=64,
                         target_modules=lora_target_modules, lora_dropout=0.05,
                         bias="none")
model.add_adapter(lora_config)

#
# def generate_batches(dataset, batch_size):
#     batch = []
#     for sample in dataset:
#         if sample['text'] and sample['wav_path'] is not None:
#             batch.append(sample)
#             if len(batch) == batch_size:
#                 yield batch
#                 batch = []
#     if batch:
#         yield batch
#
#
# def clean_qwen2_output(text):
#     # Remove everything between angle brackets like <|...|>
#     text = re.sub(r"<\|.*?\|>", "", text)
#
#     # Remove everything in square brackets like [sigh], [laugh]
#     text = re.sub(r"\[.*?\]", "", text)
#
#     # Remove any lingering angle/square brackets (e.g. <sil>)
#     text = re.sub(r"[<>]", "", text)
#
#     # Normalize spacing
#     text = re.sub(r"\s+", " ", text).strip()
#
#     return text
#
#
# with torch.inference_mode():
#     # for i in range(len(dev_set)):
#
#     for batch in generate_batches(dev_data, batch_size=1):
#
#         # prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech in English:"
#         prompts = [
#             "<|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:"
#             for _ in batch
#         ]
#
#         audios = [torchaudio.load(sample['wav_path'])[0].squeeze(0).numpy() for sample in batch]
#         refs = [sample['text'] for sample in batch]
#
#         # Process as a batch
#         inputs = processor(
#             text=prompts,
#             audio=audios,
#             return_tensors="pt",
#             add_special_tokens=True,
#             sampling_rate=16000,
#             padding=True  # <- pads input_ids and audio
#         )
#
#         # move inputs to device
#         for key in inputs:
#             inputs[key] = inputs[key].to(device)
#
#         generated_ids = model.generate(
#             **inputs,
#             max_length=1024,
#             num_beams=5,
#             no_repeat_ngram_size=3,
#             repetition_penalty=1.2,
#             length_penalty=0.9,
#             early_stopping=True,
#             # 🔥 Remove top_k / top_p when not sampling
#             # top_k=0,
#             # top_p=0.01,
#         )
#
#         # Remove prompt tokens
#         prompt_lens = inputs["input_ids"].ne(processor.tokenizer.pad_token_id).sum(dim=1)
#         clean_outputs = [
#             gen_ids[prompt_len:] for gen_ids, prompt_len in zip(generated_ids, prompt_lens)
#         ]
#
#         generated_ids = generated_ids[:, inputs.input_ids.size(1):]
#
#         # Decode
#         responses = processor.batch_decode(
#             clean_outputs,
#             skip_cspecial_tokens=True,
#             clean_up_tokenization_spaces=True
#         )
#
#         def strip_known_tags(text):
#             return text.replace("<|en|>", "").replace("<|endoftext|>", "").strip()
#
#         responses = [clean_qwen2_output(strip_known_tags(d)) for d in responses]
#
#         for hypo, ref in zip(responses, refs):
#             print("Hypo:", hypo)
#             print("Ref: ", ref)
#             print("")


from transformers import get_inverse_sqrt_schedule
from transformers import Seq2SeqTrainingArguments

learning_rate = 0.001  # args.learning_rate
warmup_steps = 2000  # args.warmup_steps

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=0.00001  # 0.0005
)

lr_scheduler = get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=warmup_steps)

output_dir = "../toydata/tmp"
batch_size = 1
gradient_accumulation = 1
max_steps = 1000

log_dir = os.path.join(output_dir, "logs")
# TODO: logging_dir
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    logging_dir=log_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,  # 1e-3,#5e-5,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    ddp_find_unused_parameters=False,
    num_train_epochs=100,
    gradient_checkpointing=False,
    bf16=True,
    length_column_name="duration",
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=1,
    save_steps=100,  # args.save_steps,
    eval_steps=100,  # args.eval_steps,
    logging_steps=10,  # args.logging_steps,
    eval_accumulation_steps=100,
    dataloader_num_workers=1,
    per_device_eval_batch_size=32,
    dataloader_persistent_workers=False,
    label_smoothing_factor=0,  # 0.1,
    load_best_model_at_end=True,
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    disable_tqdm=False,
    save_safetensors=True,
    metric_for_best_model="eval_loss",  # Must match the key in the metrics dict
    report_to="none"
)

data_collator = DataCollatorForQwen2(processor,
                                     prompt_template="<|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:",
                                     eos_token="<|endoftext|>")

callbacks = list()
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=dev_data,
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler),
    # compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=callbacks,

)

trainer.train(resume_from_checkpoint=False)
