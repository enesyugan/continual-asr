import os
import sys
import re
import warnings
import argparse

from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets, \
    interleave_datasets
from pydub import AudioSegment

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, SeamlessM4TForSpeechToText, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
from transformers import Seq2SeqTrainingArguments, SpeechEncoderDecoderConfig, AutoFeatureExtractor, WhisperConfig
from transformers import Seq2SeqTrainer, TrainerCallback
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

from trainers.trainer_shuffle import MemSeq2SeqTrainer

import random
import copy
import torch
import numpy as np
from typing import Any, Dict, List, Union
from decimal import Decimal, getcontext
from transformers import get_inverse_sqrt_schedule
from torch.nn.utils.rnn import pad_sequence
from torch import nn

from memory_efficient_whisper import create_whisper_model
from utils import DataCollatorSpeechSeq2SeqWithPadding

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

if local_rank != 0:
    # Suppress stdout and stderr for non-zero ranks
    sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, "w")
    warnings.filterwarnings("ignore")  # Ignore all warnings

parser = argparse.ArgumentParser(description='create_dataset_whisper')
parser.add_argument('-model_size', type=str, default="small",
                    help='Whisper Model size: ["large", "small".')
parser.add_argument('-low_rank_type', type=str, default="lora",
                    help='Whisper Model size: ["lora", "pissa", "olora", "eva", "rslora", "dora"')
parser.add_argument('-low_rank_modules', type=str, default="qv",
                    help='Whisper Model size: ["qv", "all-linear"')
parser.add_argument('-dataset', required=True,
                    help="Path to the dataset in huggingface format")
parser.add_argument('-checkpoint_path', default="",
                    help="Path to model checkpoint to be loaded")
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Peak learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-warmup_steps', type=int, default=2000,
                    help='Number of warm up steps for learning rate')
parser.add_argument('-lr_scheduler', type=str, default="inv_sqrt",
                    help='LR scheduler: ["inv_sqrt", "cosine".')
parser.add_argument('-spec_augment', action='store_true',
                    help="Use spec augmentation")
parser.add_argument('-label_smoothing', type=float, default=0.0,
                    help="""Label smoothing""")
parser.add_argument('-no_progress_bar', action='store_true',
                    help="Use spec augmentation")
parser.add_argument('-batch_size', type=int, default=8,
                    help='Batch size during training (per device)')
parser.add_argument('-gradient_accumulation', type=int, default=2,
                    help='Number of gradient accumulation steps')

parser.add_argument('-attn_implementation', type=str, default="flash_attention_2",
                    help='Whisper Model size: ["flash_attention_2", "sdpa", "manual"')
args = parser.parse_args()

# TODO: add option to
if args.dataset.lower() == 'seame':
    from prepare_data import get_data_seame

    all_tr_dataset, all_dev_dataset = get_data_seame(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
elif args.dataset.lower() == 'fisher':
    from prepare_data import get_data_fisher

    all_tr_dataset, all_dev_dataset = get_data_fisher(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
elif args.dataset.lower() == 'arzen':
    from prepare_data import get_data_arzen

    all_tr_dataset, all_dev_dataset = get_data_arzen(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
elif args.dataset.lower() == 'ascend':
    from prepare_data import get_data_ascend

    all_tr_dataset, all_dev_dataset = get_data_ascend(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
elif args.dataset.lower() == 'talcs':
    from prepare_data import get_data_talcs

    all_tr_dataset, all_dev_dataset = get_data_talcs(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
elif args.dataset.lower() == 'tunswitch':
    from prepare_data import get_data_tunswitch

    all_tr_dataset, all_dev_dataset = get_data_tunswitch(debug=False)
    print("Training data: {}".format(all_tr_dataset))
    print("DEV data: {}".format(all_dev_dataset))
    training_uid_mapper = None
    dev_uid_mapper = None
else:
    raise NotImplementedError("Unknown dataset: {}".format(args.dataset))


# training_uid_mapper = {key: idx for idx, key in enumerate(concat_tr_dataset["uid"])}
# dev_uid_mapper = {key: idx for idx, key in enumerate(all_dev_dataset["uid"])}


def count_parameters(model: nn.Module):
    total_params = 0
    frozen_params = 0

    for param in model.parameters():
        num_params = param.numel()  # Total number of elements (parameters)
        total_params += num_params
        if not param.requires_grad:
            frozen_params += num_params
    print(f"Total parameters: {total_params}")
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {total_params - frozen_params}")

    return total_params, frozen_params


device = device if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# model_name = "openai/whisper-large-v3-turbo"
# model_
if args.model_size == "large":
    model_name = "openai/whisper-large-v3-turbo"
else:
    model_name = "openai/whisper-small"

checkpoint_path = model_name if args.checkpoint_path == "" else args.checkpoint_path

processor = AutoProcessor.from_pretrained(model_name)

model = create_whisper_model(checkpoint_path, torch_dtype,
                             attn_implementation=args.attn_implementation,
                             low_cpu_mem_usage=True,
                             device_map={"": device})

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.label_smoothing = args.label_smoothing
print("pad_token_id: {}".format(model.config.pad_token_id))

print(model)

if args.low_rank_modules == "qv":

    lora_target_modules = ["q_proj", "v_proj"]

elif args.low_rank_modules == "all-linear":

    lora_target_modules = "all-linear"

else:
    raise NotImplementedError

if args.low_rank_type == "lora":
    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules, lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])
elif args.low_rank_type == "pissa":
    """
    PiSSA initializes the LoRA adapter using the principal singular values and singular vectors
    """
    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                             init_lora_weights="pissa",
                             lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])

elif args.low_rank_type == "olora":
    """
    Olora: QR decomposition to initialize the LoRA adapters. OLoRA translates the base weights of the model by a factor of their QR decompositions, 
    i.e., it mutates the weights before performing any training on them
    """

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                             init_lora_weights="olora",
                             lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])

elif args.low_rank_type == "eva":

    """
    EVA performs SVD on the input activations of each layer and uses the right-singular vectors to initialize LoRA weights
    """
    from peft import EvaConfig

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                             init_lora_weights="eva",
                             eva_config=EvaConfig(rho=2.0),
                             lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])

elif args.low_rank_type == "rslora":

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                             use_rslora=True,
                             lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])

elif args.low_rank_type == "dora":
    """
    decomposes the updates of the weights into two parts, magnitude and direction. 
    Direction is handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter
    """

    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                             use_dora=True,
                             lora_dropout=0.05,
                             bias="none")  # , modules_to_save=["pre_proj_out"])

else:
    raise NotImplementedError

model.add_adapter(lora_config)

count_parameters(model)

# learning_rate = 1e-3
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=0.0005
)

# lr_scheduler = get_inverse_sqrt_schedule(
#     optimizer=optimizer,
#     num_warmup_steps=warmup_steps,
# )

if args.lr_scheduler in ['inv_sqrt', 'noam']:
    lr_scheduler = get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=warmup_steps)

elif args.lr_scheduler == 'cosine':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_steps)

else:
    raise NotImplementedError

output_dir = "outputs/model_%s_%s_%s_%s" % (args.model_size, args.dataset, args.low_rank_type, args.low_rank_modules)

# TODO: logging_dir
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    # logging_dir="/export/data1/data/eugan/ASR/model/DE.EN.AR.UA.ES.ZH.TR.JA/whisper.v3/log",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,  # 1e-3,#5e-5,
    warmup_steps=warmup_steps,
    # max_steps=180000,
    ddp_find_unused_parameters=False,
    num_train_epochs=100,
    gradient_checkpointing=False,
    bf16=True,
    # group_by_length=True,
    length_column_name="duration",
    # optim="adafactor",
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=1,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    eval_accumulation_steps=100,
    dataloader_num_workers=4,
    per_device_eval_batch_size=32,
    dataloader_persistent_workers=False,
    label_smoothing_factor=0,  # 0.1,
    #   dataloader_prefetch_factor=2,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    # metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    disable_tqdm=args.no_progress_bar,
    # push_to_hub=True,
)

print("all_tr_dataset: {}".format(all_tr_dataset))
# print(type(all_tr_dataset))
getcontext().prec = 50
probabilities = list()
for _ in range(len(all_tr_dataset)):
    probability = Decimal(1) / Decimal(len(all_tr_dataset))
    probabilities.append(probability)
train_dataset = interleave_datasets(list(all_tr_dataset.values()), probabilities, seed=42)
# print("TTTTT: {}".format(train_dataset))

data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=processor.feature_extractor,
                                                     text_processor=processor.tokenizer, model_config=model.config,
                                                     uid_mapper=training_uid_mapper, dataset=train_dataset,
                                                     do_augment=args.spec_augment)

eval_data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=processor.feature_extractor,
                                                          text_processor=processor.tokenizer, model_config=model.config,
                                                          uid_mapper=dev_uid_mapper, dataset=all_dev_dataset,
                                                          do_augment=False)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10)


class LoadFullModelCallback(TrainerCallback):
    def on_train_end(self, args, state, control, model=None, **kwargs):
        if args.load_best_model_at_end and state.best_model_checkpoint:
            print(f"Loading best model from {state.best_model_checkpoint}")
            model = PeftModel.from_pretrained(model, state.best_model_checkpoint)
            model.save_pretrained(state.best_model_checkpoint, save_adapter=False)


# trainer = Seq2SeqTrainer(
trainer = MemSeq2SeqTrainer(
    train_dataset_dict=all_tr_dataset,
    eval_data_collator=eval_data_collator,
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=all_dev_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler),
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[early_stopping, LoadFullModelCallback()]
)

# trainer.state.stateful_callbacks['EarlyStoppingCallback'] = early_stopping

trainer.train(resume_from_checkpoint=False)

# if _model == model_name:
#    print("SAME")
#    trainer.train(resume_from_checkpoint=False)
# else:
#    print("NOT SAME")
#    trainer.train(resume_from_checkpoint=False)
