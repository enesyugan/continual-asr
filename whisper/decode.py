import copy

import evaluate
import sacrebleu
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch.multiprocessing as mp
from peft import PeftModel

mp.set_start_method('spawn', force=True)
import os
import signal
from tqdm import tqdm

# from sklearn.metrics import classification_report
# from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets
# from pydub import AudioSegment
# import sacrebleu
# from peft import PeftModel, PeftConfig
# from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
# from torch import nn

# import re
import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
from torch.utils.data import DataLoader, Subset
from jiwer import wer, cer
import jiwer

import torch

import sys
import argparse

from decode_utils import (DataCollatorSpeechSeq2SeqWithPadding,
                          load_asr_dataset,
                          compute_metrics,
                          remove_special_characters)

from memory_efficient_whisper import create_whisper_model


def split_dataset(dataset, num_chunks):
    """Splits dataset into `num_chunks` evenly."""
    num_samples = len(dataset)
    indices = torch.arange(num_samples)
    chunk_size = (num_samples + num_chunks - 1) // num_chunks
    return [Subset(dataset, indices[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]


# def create_dataloaders(dataset, num_chunks, batch_size):
#     """Creates chunked DataLoaders."""
#     dataset_chunks = split_dataset(dataset, num_chunks)
#     return [DataLoader(chunk, batch_size=batch_size, shuffle=False, num_workers=4) for chunk in dataset_chunks]


def load_model_and_decode(rank, dataset_split, model_path, lora_path,
                          device_id, batch_size, beam_size, total_samples,
                          no_progress_bar, lora_weights, result_queue):
    """Loads model on specific GPU and decodes its chunk."""
    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{rank}")
    device = device if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = create_whisper_model(model_path, torch_dtype,
                                 attn_implementation="flash_attention_2",
                                 low_cpu_mem_usage=True,
                                 device_map={"": device})

    if lora_path is not None and len(lora_path) > 0:

        lora_paths = lora_path.split("|")
        main_model = model

        print("[INFO] Loading LORA weights from {}".format(lora_paths[0]))
        main_model = PeftModel.from_pretrained(main_model, lora_paths[0])
        main_model.merge_and_unload()
        # checkpoint = main_model.state_dict()

        if len(lora_weights) > 0:
            lora_weights = [float(x) for x in lora_weights.split("|")]

            assert len(lora_weights) >= len(lora_paths)
            lora_weights = lora_weights[:len(lora_paths)]

            _sum = sum(lora_weights)
            for i in range(len(lora_weights)):
                lora_weights[i] = lora_weights[i] / _sum
        else:
            lora_weights = [1/len(lora_paths)] * len(lora_paths)

        if len(lora_paths) > 1:

            # averaging the parameters

            for _idx, _lora_path in enumerate(lora_paths[1:]):
                new_model = create_whisper_model(model_path, torch_dtype,
                                                 attn_implementation="flash_attention_2",
                                                 low_cpu_mem_usage=True,
                                                 device_map={"": device})

                print("[INFO] Loading LORA weights from {}".format(_lora_path))

                new_model = PeftModel.from_pretrained(new_model, _lora_path)
                new_model.merge_and_unload()

                for (main_param, param) in zip(main_model.parameters(), new_model.parameters()):

                    if _idx == 0:
                        main_param.data.mul_(lora_weights[0]).add_(param.data.mul_(lora_weights[1 + _idx]))
                    else:
                        main_param.data.add_(param.data.mul_(lora_weights[1 + _idx]))

            # for main_param in main_model.parameters():
            #     main_param.data.div_(len(lora_paths))

        model = main_model

    upstream_model = "openai/whisper-large-v3-turbo"

    processor = AutoProcessor.from_pretrained(upstream_model)

    zh_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    en_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
    es_id = processor.tokenizer.convert_tokens_to_ids("<|es|>")
    de_id = processor.tokenizer.convert_tokens_to_ids("<|de|>")

    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    # print(f"zh: {zh_id} en: {en_id} es: {es_id} de: {de_id}")
    forced_decoder_ids = list()
    # print(model.generation_config.forced_decoder_ids[0], flush=True)
    # print(ASD)
    forced_decoder_ids.append(model.generation_config.forced_decoder_ids[0])
    forced_decoder_ids.append([2, transcribe_id])
    forced_decoder_ids.append([3, notimestamps_id])

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # model = torch.compile(model)
    model.cuda()
    model.eval()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, text_processor=processor.tokenizer)

    data_loader = DataLoader(
        dataset=dataset_split,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False)

    def to_cuda_recursive(data, device, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.to(device, dtype=dtype)
        elif isinstance(data, list):
            return [to_cuda_recursive(item, device, dtype=dtype) for item in data]
        elif isinstance(data, dict):
            return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
        elif isinstance(data, transformers.feature_extraction_utils.BatchFeature):
            return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
        else:
            return data

    target_lst = list()
    predictions_lst = list()
    target_language_lst = list()

    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.num_return_sequences = 1
    model.generation_config.num_beams = beam_size
    model.generation_config.no_repeat_ngram_size = 4
    model.generation_config.max_new_tokens = 255

    language_tokens = [t for t in processor.tokenizer.additional_special_tokens if len(t) == 6]
    language_ids = [processor.tokenizer.convert_tokens_to_ids(x) for x in language_tokens]

    if rank == 0:
        print(language_tokens)
        print(language_ids)

    # if not no_progress_bar:
    #     progress_bar = tqdm(total=total_samples, position=rank, desc=f"GPU {device_id}", leave=True)
    # else:
    #     progress_bar = None

    with torch.no_grad(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, batch_ in tqdm(enumerate(data_loader), total=len(data_loader), position=rank,
                                desc=f"GPU {device_id}", leave=True,
                                disable=no_progress_bar):
            path_lst = batch_.pop("paths")
            target = batch_.pop("tgt_txt")
            uid = batch_.pop("uid")
            language = batch_.pop("lang_labels")
            # target_language_lst.extend(processor.batch_decode(language, skip_special_tokens=False))
            # print(batch_)
            for t in target:
                t = t.split(" ", 1)[-1].replace("<|endoftext|>", "")
                target_lst.append(t)
            # print(target_lst)
            # target_lst.extend(target)
            batch = to_cuda_recursive(batch_, device, torch_dtype)

            output_tokens = model.generate(input_features=batch["input_features"],
                                           generation_config=model.generation_config)
            # forced_decoder_ids=forced_decoder_ids, num_return_sequences=1, num_beams=10, no_repeat_ngram_size=4,
            # max_new_tokens=255)#, task="transcribe") print(output_tokens)
            pred_transcript = processor.batch_decode(output_tokens, skip_special_tokens=True)
            # print(uid)
            predictions_lst.extend(pred_transcript)

            # if idx < 3:
            #     print(output_tokens)
            #     print(f"target: {target_lst}")
            #     # print(f"target_language_lst: {target_language_lst}")
            #     print(f"pred_language: {predictions_lst}")
            # else:
            #     continue

    results = list(zip(predictions_lst, target_lst))

    result_queue.put(results)


if __name__ == "__main__":

    def handle_sigint(sig, frame):
        print("\nReceived Ctrl+C, terminating all processes...")
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_sigint)  # Handle Ctrl+C globally

    parser = argparse.ArgumentParser(description='translate_whisper.py')

    parser.add_argument('-model_path', required=True, default="", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('-lora_path', required=False, default="", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('-test_stm', required=True, default="test_length.cl_lc.stm",
                        help='Source file to decode (one line per sequence)')
    # parser.add_argument('-huggingface_dataset', required=False, default="",
    #                     help="If src is none, using huggingface dataset")
    parser.add_argument('-batch_size', type=int, default=8,
                        help='Batch size during decoding ')
    parser.add_argument('-beam_size', type=int, default=4,
                        help='Beam size during decoding')
    parser.add_argument('-output_file', required=False, default="",
                        help="Path to the output_file to be written")
    parser.add_argument('-target_file', required=False, default="",
                        help="Path to the reference file. If provided word error rate will be computed")

    parser.add_argument('-no_progress_bar', action='store_true',
                        help="Use spec augmentation")

    parser.add_argument('-lora_weights', required=False, default="", type=str,
                        help="Efficients for each lora set")

    args = parser.parse_args()

    test_path = args.test_stm

    # for arzen it should be ar en
    test_dataset = load_asr_dataset(test_path, language=["ar", "en"])
    print(test_dataset)

    num_gpus = torch.cuda.device_count()
    dataset_chunks = split_dataset(test_dataset, num_gpus)

    # Shared queue for results
    result_queue = mp.Queue()

    # Spawn processes
    total_size = len(test_dataset)

    if num_gpus > 1:

        processes = []
        for gpu_id in range(num_gpus):
            process = mp.Process(target=load_model_and_decode,
                                 args=(gpu_id, dataset_chunks[gpu_id], args.model_path, args.lora_path,
                                       gpu_id, args.batch_size, args.beam_size, total_size,
                                       args.no_progress_bar, args.lora_weights,
                                       result_queue))
            process.start()
            processes.append(process)

    else:
        load_model_and_decode(0, test_dataset, args.model_path, args.lora_path,
                              0, args.batch_size, args.beam_size, total_size,
                              args.no_progress_bar, args.lora_weights, result_queue)

    # Collect results
    final_results = []
    for _ in range(num_gpus):
        final_results.extend(result_queue.get())

    predictions_lst = list()
    target_lst = list()

    for decoding_output in final_results:
        predictions_lst.append(decoding_output[0])
        target_lst.append(decoding_output[1])

    clean_predictions = list()
    for text in predictions_lst:
        text = remove_special_characters(text)
        clean_predictions.append(text)

    outdir = os.path.basename(args.test_stm).replace(".stm", "")
    os.makedirs(outdir, exist_ok=True)

    # TODO: having proper folder and output filenames
    with open(os.path.join(outdir, "hypos.txt"), "w") as f:
        for line in predictions_lst:
            f.write("{}\n".format(line.strip()))

    with open(os.path.join(outdir, "target.txt"), "w") as f:
        for line in target_lst:
            f.write("{}\n".format(line.strip()))

    with open(os.path.join(outdir, "hypos.norm.txt"), "w") as f:
        for line in clean_predictions:
            f.write("{}\n".format(line.strip()))

    wer_error = wer(target_lst, clean_predictions)
    cer_error = cer(target_lst, clean_predictions)

    with open(os.path.join(outdir, "error_rates.txt"), "w") as f:

        f.write("WER: {}\n".format(wer_error))
        f.write("CER: {}\n".format(cer_error))
        print("WER: {}".format(wer_error))
        print("CER: {}".format(cer_error))

    out = jiwer.process_words(target_lst,
                              clean_predictions)

    # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
    with open(os.path.join(outdir, "w.eval.txt"), "w") as f:
        f.write(jiwer.visualize_alignment(out))

    out = jiwer.process_characters(target_lst,
                                   clean_predictions)
    # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
    with open(os.path.join(outdir, "c.eval.txt"), "w") as f:
        f.write(jiwer.visualize_alignment(out))
