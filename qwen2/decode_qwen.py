from pathlib import Path
import re
import json
import argparse
from transformers import AutoProcessor
from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
import torch.multiprocessing as mp
from collator import EvalDataCollatorForQwen2
from tqdm import tqdm
import jiwer
from jiwer import wer, cer

mp.set_start_method('spawn', force=True)
import os
import signal
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from qwen2.qwen2model import create_qwen2audio_model
from whisper.decode_utils import remove_special_characters

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qwen2.models.qwen2audio_attention import Qwen2AudioFlashAttentionNoPad
from loras.bnn_lora import BLoBConfig, BLoB, BLoBModel  # BayesianLinear, BayesianLoraConfig
from peft import PeftModel, LoraModel

# from transformers.models.qwen2_audio.modeling_qwen2_audio import QWEN2AUDIO_ATTENTION_CLASSES

# test command:
# python -m qwen2.test_qwen

# Inject your class into the registry
# QWEN2AUDIO_ATTENTION_CLASSES["flash_attention_2"] = Qwen2AudioFlashAttentionNoPad
# QWEN2AUDIO_ATTENTION_CLASSES["eager"] = Qwen2AudioAttentionNoMask

# attention_type = "eager"
attention_type = "flash_attention_2"


def find_weight_path(weight_path, auto_find_checkpoint="none"):
    if auto_find_checkpoint == "none":
        return weight_path

    base_path = Path(weight_path)

    ckpt_dirs = []

    for entry in base_path.iterdir():
        if entry.is_dir():
            match = re.match(r"checkpoint-(\d+)", entry.name)
            if match:
                step = int(match.group(1))
                ckpt_dirs.append((step, entry))

    best_ckpt = None
    best_loss = float("inf")

    # print(weight_path)
    checkpoints = sorted(ckpt_dirs)

    # print(checkpoints)

    for step, path in checkpoints:
        eval_file = path / "eval_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                loss = data.get("eval_loss")
                if loss is not None and loss < best_loss:
                    best_loss = loss
                    best_ckpt = path

    if best_ckpt is not None and auto_find_checkpoint == "best":
        return best_ckpt
    else:
        earliest_ckpt = checkpoints[0][1] if checkpoints else None
        latest_ckpt = checkpoints[-1][1] if checkpoints else None

        assert earliest_ckpt is not None, "Cannot find any checkpoint!!!"
        assert latest_ckpt is not None, "Cannot find any checkpoint!!!"

        if auto_find_checkpoint == "latest":
            return latest_ckpt
        elif auto_find_checkpoint == "best":
            return earliest_ckpt


def split_dataset(dataset, num_chunks):
    """Splits dataset into `num_chunks` evenly."""
    num_samples = len(dataset)
    indices = torch.arange(num_samples)
    chunk_size = (num_samples + num_chunks - 1) // num_chunks
    return [Subset(dataset, indices[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]


def load_model_and_decode(rank, dataset_split, model_path, lora_path, auto_find_checkpoint, tokenizer_path,
                          tgt_lang, custom_lora, device_id, batch_size, beam_size,
                          no_repeat_ngram_size, total_samples,
                          no_progress_bar, lora_weights, result_queue):
    def pprint(*args, **kwargs):
        if rank > 0:
            return

        print(*args, **kwargs)

    """Loads model on specific GPU and decodes its chunk."""
    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{rank}")
    device = device if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    config = AutoConfig.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

    model = create_qwen2audio_model("Qwen/Qwen2-Audio-7B", config,
                                    torch_dtype=torch_dtype,
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True,
                                    attn_implementation=attention_type,
                                    mem_efficient=True,
                                    device_map={"": device})

    if lora_path is not None and len(lora_path) > 0:

        lora_paths = lora_path.split("|")
        main_model = model

        # print(lora_paths)
        weight_path = str(find_weight_path(lora_paths[0], auto_find_checkpoint))

        pprint("[INFO] Loading LORA weights from {}".format(weight_path))
        if custom_lora:

            LoraModel._create_and_replace = BLoBModel._create_and_replace

            lora_config = BLoBConfig.from_pretrained(weight_path)
            pprint(lora_config)
            lora_config._register_custom_module({nn.Linear: BLoB})
            main_model = PeftModel.from_pretrained(main_model, model_id=weight_path, config=lora_config)

            main_model.merge_and_unload()
        else:
            main_model = PeftModel.from_pretrained(main_model, weight_path)
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
            lora_weights = [1 / len(lora_paths)] * len(lora_paths)

        if len(lora_paths) > 1:

            # averaging the parameters

            for _idx, _lora_path in enumerate(lora_paths[1:]):
                new_model = create_qwen2audio_model("Qwen/Qwen2-Audio-7B", config,
                                                    torch_dtype=torch_dtype,
                                                    trust_remote_code=True,
                                                    low_cpu_mem_usage=True,
                                                    attn_implementation=attention_type,
                                                    mem_efficient=True,
                                                    device_map={"": device})

                _weight_path = str(find_weight_path(_lora_path, auto_find_checkpoint))
                pprint("[INFO] Loading LORA weights from {}".format(_weight_path))

                if custom_lora:

                    LoraModel._create_and_replace = BLoBModel._create_and_replace

                    lora_config = BLoBConfig.from_pretrained(_weight_path)
                    print(lora_config)
                    lora_config._register_custom_module({nn.Linear: BLoB})
                    new_model = PeftModel.from_pretrained(new_model, model_id=_weight_path, config=lora_config)

                    new_model.merge_and_unload()
                else:
                    new_model = PeftModel.from_pretrained(new_model, _weight_path)
                    new_model.merge_and_unload()

                # new_model = PeftModel.from_pretrained(new_model, _weight_path)
                # new_model.merge_and_unload()

                for (main_param, param) in zip(main_model.parameters(), new_model.parameters()):

                    if _idx == 0:
                        main_param.data.mul_(lora_weights[0]).add_(param.data.mul_(lora_weights[1 + _idx]))
                    else:
                        main_param.data.add_(param.data.mul_(lora_weights[1 + _idx]))

            # for main_param in main_model.parameters():
            #     main_param.data.div_(len(lora_paths))

        model = main_model

    if len(tokenizer_path) > 0:
        upstream_model = tokenizer_path  # "openai/whisper-large-v3-turbo"
    else:
        upstream_model = model_path

    processor = AutoProcessor.from_pretrained(upstream_model)

    model.cuda()
    model.eval()

    data_collator = EvalDataCollatorForQwen2(processor,
                                             prompt_template="<|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:",
                                             eos_token="<|endoftext|>")

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
        elif isinstance(data, BatchFeature):
            return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
        else:
            return data

    target_lst = list()
    predictions_lst = list()
    target_language_lst = list()

    with torch.no_grad(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, batch_ in tqdm(enumerate(data_loader), total=len(data_loader), position=rank,
                                desc=f"GPU {device_id}", leave=True,
                                disable=no_progress_bar):

            audios = batch_["audio"]
            refs = batch_["text"]

            prompts = [
                "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
                for _ in audios
            ]

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
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
                # repetition_penalty=1.2,
                # length_penalty=0.9,
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

                for tag in ["<|en|>", "<|endoftext|>", "<|AUDIO|>", "<|audio_eos|>", "[noise]", "<|audio_bos|>"]:
                    text = text.replace(tag, "")

                text = text.strip()

                if text.startswith("Transcribe this speech:"):
                    text = text[len("Transcribe this speech:"):]

                def remove_bracket_tags(text):
                    # Removes [anything in square brackets], including the brackets
                    return re.sub(r'\[[^\]]*\]', '', text).strip()

                text = remove_bracket_tags(text)

                def remove_language_tags(text):
                    return re.sub(r'<\|[a-zA-Z_-]+?\|>', '', text)

                text = remove_language_tags(text)

                return text

            pred_transcript = [strip_known_tags(d) for d in responses]
            predictions_lst.extend(pred_transcript)
            target_lst.extend(refs)

            if device_id == 0:
                for i in range(len(refs)):
                    print(f"[REF]: {refs[i]}")
                    print(f"[HYP]: {pred_transcript[i]}")
                    print("")

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
    parser.add_argument('-tokenizer_path', required=False, default="", type=str,
                        help="Path to the tokenizer (provided if the model checkpoint doesn't have)")
    parser.add_argument('-lora_path', required=False, default="", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('-test_stm', required=True, default="test_length.cl_lc.stm",
                        help='Source file to decode (one line per sequence)')
    parser.add_argument('-tgt_lang', required=False, default="", type=str,
                        help="set language token to decode into,i.e. <|en|>, <|de|>, ...")
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

    parser.add_argument('-custom_lora', action='store_true',
                        help="Use spec augmentation")

    parser.add_argument('-lora_weights', required=False, default="", type=str,
                        help="Efficients for each lora set")

    parser.add_argument('-keep_special_character', action='store_true',
                        help="Ignore the special character removal")

    parser.add_argument('-no_repeat_ngram_size', type=int, default=4,
                        help='Prevent ngram repetition with this size.')

    parser.add_argument('-no_progress_bar', action='store_true',
                        help="Disable the progress bar")

    parser.add_argument('-auto_find_checkpoint', default="none", type=str,
                        help="Automatically find checkpoint to easily use with huggingface training/tuning. Options: none|best|latest")

    args = parser.parse_args()
    args.no_progress_bar = True

    test_path = args.test_stm

    from qwen2.stm_to_dataset import load_stm_file

    test_dataset = load_stm_file(test_path, lower=True, remove_punct=True, special_char_removal=True)

    print(test_dataset)

    num_gpus = torch.cuda.device_count()
    dataset_chunks = split_dataset(test_dataset, num_gpus)

    # Shared queue for results
    result_queue = mp.Queue()

    # Spawn processes
    total_size = len(test_dataset)

    # weight_path = str(find_weight_path(args.lora_path, args.auto_find_checkpoint))
    # print("Using checkpoint:", weight_path)
    weight_path = args.lora_path

    if num_gpus > 1:

        processes = []
        for gpu_id in range(num_gpus):
            process = mp.Process(target=load_model_and_decode,
                                 args=(gpu_id, dataset_chunks[gpu_id],
                                       args.model_path, weight_path, args.auto_find_checkpoint, args.tokenizer_path,
                                       args.tgt_lang, args.custom_lora, gpu_id,
                                       args.batch_size, args.beam_size, args.no_repeat_ngram_size,
                                       total_size, args.no_progress_bar, args.lora_weights,
                                       result_queue))
            process.start()
            processes.append(process)

    else:
        load_model_and_decode(0, test_dataset, args.model_path, weight_path, args.auto_find_checkpoint,
                              args.tokenizer_path,
                              args.tgt_lang, args.custom_lora, 0,
                              args.batch_size, args.beam_size, args.no_repeat_ngram_size, total_size,
                              args.no_progress_bar, args.lora_weights, result_queue)

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
        if not args.keep_special_character:
            text = remove_special_characters(text).strip()
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

    # ref_str = " ".join(target_lst)
    # hyp_str = " ".join(clean_predictions)
    #
    # wer_score = jiwer.wer(ref_str, hyp_str)
    # cer_score = jiwer.cer(ref_str, hyp_str)

    with open(os.path.join(outdir, "error_rates.txt"), "w") as f:

        f.write("WER: {}\n".format(wer_error))
        f.write("CER: {}\n".format(cer_error))
        print("WER: {}".format(wer_error))
        print("CER: {}".format(cer_error))

    # out = jiwer.process_words(target_lst,
    #                           clean_predictions)
    #
    # # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
    # with open(os.path.join(outdir, "w.eval.txt"), "w") as f:
    #     f.write(jiwer.visualize_alignment(out))
    #
    # out = jiwer.process_characters(target_lst,
    #                                clean_predictions)
    # # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
    # with open(os.path.join(outdir, "c.eval.txt"), "w") as f:
    #     f.write(jiwer.visualize_alignment(out))