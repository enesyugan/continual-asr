import os
import sys
import re
import warnings
import argparse
import math
from collections import defaultdict
import yaml

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
from prepare_data import get_train_dev

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
parser.add_argument('-data_config', type=str, required=True,
                    help="Path to the dataset config fule (YAML or JSON).")
parser.add_argument('-lower_case', action='store_true',
                    help="set if you want to lower case transcripts")
parser.add_argument('-output_dir', default="outputs",
                    help="Path to model checkpoint to be loaded")
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
                    help="disable the progress bar (to use with slurm)")
parser.add_argument('-batch_size', type=int, default=8,
                    help='Batch size during training (per device)')
parser.add_argument('-max_steps', type=int, default=30000,
                    help='Max number of update steps')
parser.add_argument('-save_steps', type=int, default=100,
                    help='Number of steps per saving a checkpoint')
parser.add_argument('-eval_steps', type=int, default=100,
                    help='Number of steps per evaluation on the dev set')
parser.add_argument('-logging_steps', type=int, default=10,
                    help='Number of steps per evaluation on the dev set')
parser.add_argument('-gradient_accumulation', type=int, default=2,
                    help='Number of gradient accumulation steps')

parser.add_argument('-attn_implementation', type=str, default="flash_attention_2",
                    help='Whisper Model size: ["flash_attention_2", "sdpa", "manual"')
parser.add_argument('-weight_decay', type=float, default=0.0005,
                    help="""Label smoothing""")

parser.add_argument('-ema', action='store_true',
                    help="Use exponential moving average during training")
parser.add_argument('-ema_rate', type=float, default=0.9
                    , help='Dropout value of the model')

parser.add_argument('-teacher_distillation', type=float, default=0
                    , help='Use the original Whisper model as a teacher')

parser.add_argument('-optim', type=str, default="adam",
                    help='Optimizer: ["adam", "rmsprop", "sgd".')

parser.add_argument('-freeze_embedding', action='store_true',
                    help="Use exponential moving average during training")

parser.add_argument('-disable_safetensors', action='store_true',
                    help="Use exponential moving average during training")

parser.add_argument('-keep_special_character', action='store_true',
                        help="Ignore the special character removal")

args = parser.parse_args()
print(args)


training_uid_mapper = None
dev_uid_mapper = None

# 2. Load config from YAML (or JSON)
with open(args.data_config, "r") as f:
    data_config = yaml.safe_load(f)

# 3. Call get_train_dev with the loaded config
all_tr_dataset, all_dev_dataset = get_train_dev(data_config,
                                                special_char_removal= not args.keep_special_character)

print("Training dataset loaded:", all_tr_dataset)
print("Development dataset loaded:", all_dev_dataset)

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
elif args.model_size == "small":
    model_name = "openai/whisper-small"
else:
    model_name = args.model_size

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

if args.freeze_embedding:
    model.proj_out.weight.requires_grad = False
    if model.proj_out.bias is not None:
        model.proj_out.bias.requires_grad = False
    model.model.decoder.embed_tokens.weight.requires_grad = False

if args.low_rank_modules == "qv":

    lora_target_modules = ["q_proj", "v_proj"]

elif args.low_rank_modules == "all-linear":

    lora_target_modules = "all-linear"

else:
    raise NotImplementedError

if len(args.low_rank_type) > 0:
    if args.low_rank_type == "full_ft":
        """
        Full Fine-Tuning: No adapters are used, and the entire model is fine-tuned.
        """
        print("Performing full fine-tuning: No adapters will be added.")
        # Ensure that all model parameters are trainable
        for param in model.parameters():
            param.requires_grad = True

    elif args.low_rank_type == "lora":
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules, lora_dropout=0.05,
                                 bias="none")  # , modules_to_save=["pre_proj_out"])
        model.add_adapter(lora_config)

    elif args.low_rank_type == "pissa":
        """
        PiSSA initializes the LoRA adapter using the principal singular values and singular vectors
        """
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                                 init_lora_weights="pissa",
                                 lora_dropout=0.05,
                                 bias="none")  # , modules_to_save=["pre_proj_out"])
        model.add_adapter(lora_config)

    elif args.low_rank_type == "olora":
        """
        Olora: QR decomposition to initialize the LoRA adapters. OLoRA translates the base weights of the model by a factor of their QR decompositions, 
        i.e., it mutates the weights before performing any training on them
        """

        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                                 init_lora_weights="olora",
                                 lora_dropout=0.05,
                                 bias="none")  # , modules_to_save=["pre_proj_out"])
        model.add_adapter(lora_config)

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
        model.add_adapter(lora_config)

    elif args.low_rank_type == "rslora":

        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                                 use_rslora=True,
                                 lora_dropout=0.05,
                                 bias="none")  # , modules_to_save=["pre_proj_out"])
        model.add_adapter(lora_config)

    elif args.low_rank_type == "dora":
        """
        decomposes the updates of the weights into two parts, magnitude and direction. 
        Direction is handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter
        """

        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=lora_target_modules,
                                 use_dora=True,
                                 lora_dropout=0.05,
                                 bias="none")  # , modules_to_save=["pre_proj_out"])
        model.add_adapter(lora_config)

    else:
        raise NotImplementedError


else:
    # nothing happens
    model.proj_out.weight = model.model.decoder.embed_tokens.weight
    # original_save_pretrained = model.save_pretrained

    # def patched_save_pretrained(*args, **kwargs):
    #     model.model.proj_out.weight = model.model.decoder.embed_tokens.weight
    #     return original_save_pretrained(*args, **kwargs)
    #
    # model.save_pretrained = patched_save_pretrained
    # # model.save_pretrained("my_model_path")

    pass


count_parameters(model)

if args.teacher_distillation > 0:
    print("[INFO] Using the Whisper model as a teacher")
    # actually student and teacher can be the same xD
    teacher = create_whisper_model(model_name, torch_dtype, attn_implementation="flash_attention_2",
                                   low_cpu_mem_usage=True,
                                   device_map={"": device})

    # freeze the parameters for the teacher
    for param in teacher.parameters():
        param.requires_grad = False

    teacher.config.forced_decoder_ids = None
    teacher.config.suppress_tokens = []
    teacher.config.use_cache = False  # importante

    model.teacher = teacher
    model.teacher_distillation = args.teacher_distillation

else:
    teacher = None

# learning_rate = 1e-3
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps

if args.optim in ['adam', 'adamw']:

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay  # 0.0005
    )

elif args.optim in ['sgd']:

    optimized_params = model.parameters()

    if args.ema:

        class EMASGD(torch.optim.SGD):

            def __init__(self, params, ema_rate, **kwargs):
                super().__init__(params, **kwargs)
                self.ema_rate = ema_rate
                self.counter = 0
                # self.shadow_weights = {id(p): p.clone().detach() for group in self.param_groups for p in group[
                # 'params']}
                self.shadow_weights = dict()

                for group in self.param_groups:
                    for p in group['params']:
                        self.shadow_weights[id(p)] = p.data.new_zeros(p.size())
                        self.shadow_weights[id(p)].add_(p.data)

            def step(self, closure=None):
                """
                Performs a single optimization step and updates EMA weights.
                """
                # Perform the base SGD step
                loss = super().step(closure)
                self.counter += 1

                # alpha is the weight assigned to the current parameters
                # higher = learning, lower = not learning

                # at the start of training, we want to have high alpha
                # so it should be
                # alpha = (1 / self.counter) * self.ema_rate


                # alpha = max(0.01, min(alpha, 0.5))
                # alpha = 1 - self.ema_rate

                # alpha = 1.0 * math.exp(-self.ema_rate * self.counter)
                #
                # alpha_min = 0.0001
                # alpha = alpha_min + (1.0 - alpha_min) * math.exp(-self.ema_rate * self.counter)

                # follow the same equation in the Rehearsal Free paper

                alpha = 1 / self.counter

                total = 0
                # Update EMA weights
                with torch.no_grad():
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.requires_grad:
                                param_id = id(p)
                                if param_id in self.shadow_weights:

                                    if p.device != self.shadow_weights[param_id].device:
                                        self.shadow_weights[param_id] = self.shadow_weights[param_id].to(p.device)
                                    # print("[DEBUGGING] alpha:", alpha)
                                    # print("[DEBUGGING] BEFORE")

                                    # w_sum = p.data.sum().item()
                                    # sw_sum = self.shadow_weights[param_id].sum().item()
                                    #
                                    # if w_sum != sw_sum:
                                    #     print(f"p.data sum: {w_sum}, "
                                    #           f"shadow_weights sum: {sw_sum}")
                                    p.data.mul_(alpha).add_(self.shadow_weights[param_id] * (1 - alpha))
                                    # w_sum = p.data.sum().item()
                                    # sw_sum = self.shadow_weights[param_id].sum().item()
                                    #
                                    # if w_sum != sw_sum:
                                    #     # print("[DEBUGGING] AFTER")
                                    #     print(f"p.data sum: {w_sum}, "
                                    #           f"shadow_weights sum: {sw_sum}")
                                    #
                                    #     total += p.data.numel()
                                    # print("---")

                                    self.shadow_weights[param_id].copy_(p.data)

                # print("Total number of EMA-updated params:", total)

                return loss


        optimizer = EMASGD(optimized_params, ema_rate=args.ema_rate, weight_decay=args.weight_decay,
                           lr=args.learning_rate)

    else:
        optimizer = torch.optim.SGD(model.parameters,
                                    lr=args.learning_rate)

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


#output_dir = args.output_dir + "/model_%s_%s_%s_%s" % (
#    args.model_size, "-".join(dataset_names), args.low_rank_type, args.low_rank_modules)
output_dir = args.output_dir + "/model_%s_%s_%s" % (
    args.model_size, args.low_rank_type, args.low_rank_modules)



# TODO: logging_dir
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    # logging_dir="/export/data1/data/eugan/ASR/model/DE.EN.AR.UA.ES.ZH.TR.JA/whisper.v3/log",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,  # 1e-3,#5e-5,
    warmup_steps=warmup_steps,
    max_steps=args.max_steps,
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
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
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
    save_safetensors=not args.disable_safetensors
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
    def on_train_end(self, _args, state, control, model=None, **kwargs):
        if _args.load_best_model_at_end and state.best_model_checkpoint and len(args.low_rank_type) > 0:
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
