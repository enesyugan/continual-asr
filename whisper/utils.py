from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask
import torchaudio.transforms as T
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import random
import copy
import torch
import numpy as np
from typing import Any, Dict, List, Union
from decimal import Decimal, getcontext
from transformers import get_inverse_sqrt_schedule
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import os
import sys
import re
import warnings


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    # processor: Any
    feature_extractor: Any
    text_processor: Any
    model_config: Any
    uid_mapper: Any
    dataset: Any
    do_augment: bool
    audio_augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5, leave_length_unchanged=False),
        TimeMask(min_band_part=0.0, max_band_part=0.1, p=0.5),
    ])
    spec_time_masking = T.TimeMasking(time_mask_param=30)
    spec_freq_masking = T.FrequencyMasking(freq_mask_param=30)
    
    # Arabic letters, numerals (Arabic-Indic + Extended), and punctuation (e.g. U+061F “؟”)
    _arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    # Latin letters
    _latin_re  = re.compile(r'[A-Za-z]')
    # ASCII digits
    _latin_digit_re  = re.compile(r'[0-9]')
    # Arabic-Indic digits (U+0660–U+0669) + Eastern Arabic-Indic digits (U+06F0–U+06F9)
    _arabic_digit_re = re.compile(r'[\u0660-\u0669\u06F0-\u06F9]')

    # Han (CJK Unified Ideographs) — most common Chinese characters
    _mandarin_re = re.compile(r'[\u4E00-\u9FFF]')
    # CJK-specific punctuation / fullwidth forms
    _mandarin_punct_re = re.compile(r'[\u3000-\u303F\uFF00-\uFFEF]')

    def _script_label_for_token(self, tok: str) -> int:
        if tok.startswith("<|") and tok.endswith("|>"):
            return -100

        core = tok.lstrip('ĠĊ')
        has_ar = bool(self._arabic_re.search(core)) or bool(self._arabic_digit_re.search(core))
        has_la = bool(self._latin_re.search(core))  or bool(self._latin_digit_re.search(core))
        has_zh = bool(self._mandarin_re.search(core)) or bool(self._mandarin_punct_re.search(core))

        # we now have three “flags” — ar, la, zh — so count how many scripts appear
        scripts = has_ar + has_la + has_zh
        if scripts == 1:
            if has_ar:
                return 0
            if has_la:
                return 1
            return 3

        if scripts > 1:
            return 2

        return -100

        ## 0 = Arabic-only, 1 = Latin-only, 2 = Mixed, -100 = ignore
        #if has_ar and not has_la:
        #    return 0
        #if has_la and not has_ar:
        #    return 1
        #if has_ar and has_la:
        #    return 2
        ##print(f"NO LANGUAGE: {tok}", flush=True)
        #return -100

    def init(self):
        # … your existing setup …
        tokenizer = self.text_processor
        
        # 1) collect every ID the tokenizer can ever output
        core_ids    = set(tokenizer.get_vocab().values())
        special_ids = set(tokenizer.all_special_ids)
        all_ids     = core_ids.union(special_ids)

        # 2) find the largest ID
        max_id = max(all_ids)

        # 3) build a full (-100) table of size (max_id+1,)
        id2label = torch.full((max_id + 1,), -100, dtype=torch.int64)
        
        # 4) fill in only the IDs that actually exist
        for tok_id in all_ids:
            #tok      = tokenizer.convert_ids_to_tokens(tok_id)
            tok      = tokenizer.decode([tok_id], clean_up_tokenization_spaces=True)
            if tok_id==1392 or tok_id==23032 or tok_id==118 or tok_id==50258 or tok_id==50272: 
                print(f"{tok_id}: {tok}")
                tid = tok_id -1
                t = tokenizer.decode([tid], clean_up_tokenization_spaces=True)
                print(f"{tid}: {t}")

                tid = tok_id +1
                t = tokenizer.decode([tid], clean_up_tokenization_spaces=True)
                print(f"{tid}: {t}")

            id2label[tok_id] = self._script_label_for_token(tok)
        
        # store on the device you’ll be using (cpu or cuda)
        self.id2label = id2label  # you can also call .to(device) later
        print(f"3555: {self.id2label[3555]}")
        print(f"15040: {self.id2label[15040]}")

    def get_language_for_token(self, words_batch, language_label_lst, decoder_input_ids):
        bpe_languages_batch = list()
        for idx, words in enumerate(words_batch):
            bpe_languages = ["<|transcribe|>", "<|notimestamps|>"]
            # bpe_languages = []
            words = words.split(" ", 1)[-1].replace("<|endoftext|>", "")

            languages = language_label_lst[idx]  # self.detect_language(words)
            len_ids = len(decoder_input_ids[idx])
            # print("===")
            toks = list()
            for word, language in zip(words.split(), languages):
                tok = self.text_processor(" " + word, return_tensors="pt", padding=False, add_special_tokens=False)
                #   print("word: {}".format(tok.input_ids))
                #    toks.extend(tok.input_ids)
                bpe_languages.extend([language] * tok.input_ids.shape[1])
            bpe_languages.append(bpe_languages[-1])
            # if len_ids != len(bpe_languages):
            #     print(words_batch)
            #     print(language_label_lst)
            #     print(bpe_languages)
            #     print(toks)
            #     print(decoder_input_ids)
            #     input("Press Enter to continue...")
            # print(bpe_languages)
            bpe_languages_batch.append("".join(bpe_languages))
            # bpe_languages_batch.append(torch.tensor(bpe_languages))

        # print(bpe_languages_batch, flush=True)
        padded_tensor = self.text_processor(bpe_languages_batch, return_tensors="pt", add_special_tokens=False,
                                            padding=True)
        # print(padded_tensor, flush=True)
        # print("words_batch {} ".format(words_batch))
        # print("decoder_input_ids:  {} shape {}".format(decoder_input_ids, decoder_input_ids.shape))
        # padded_tensor = pad_sequence(bpe_languages_batch, batch_first=True, padding_value=-100)
        # print("bpe_languages_batch: {} ".format(bpe_languages_batch))
        # print(padded_tensor.input_ids, flush=True);
        # print(type(padded_tensor))
        return padded_tensor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        sr = 16000  # hardcored length
        audio_lst = list()
        label_lst = list()
        num_languages_lst = list()
        language_ids_lst = list()
        lang_mix_id_lst = list()
        max_length = [250, 500, 750, 1000]
        max_duration = [5 * sr, 10 * sr, 15 * sr, 20 * sr]
        longest_audio = 0

        for el in features:
            uid = el["uid"]
            # print(el["transcript"])
            if len(uid.split("--")) > 1:
                uids = uid.split('--')
                num_languages_lst.append(len(uids))
                indexes = [self.uid_mapper[uid] for uid in uids if uid in self.uid_mapper]
                elements = [self.dataset[idx] for idx in indexes]
                audio = np.concatenate([self.dataset[idx]["audio"]["array"] for idx in indexes])

            else:
                num_languages_lst.append(1)
                audio = el["audio"]["array"]

            if len(audio) < 1:
                # print(el["audio"], flush=True)
                continue

            if not el.get("start", None) is None:
                if int(float(el["start"])) >= 0 and int(float(el["end"])) > 0:
                    start = int(el["start"])
                    end = int(el["end"])
                    audio = audio[start * 16: end * 16]

            new_audio = self.audio_augment(audio, sample_rate=16000) if self.do_augment else audio

            if len(new_audio) <= 20 * sr:
                audio = new_audio

            if len(audio) > longest_audio:
                longest_audio = len(audio)

            audio_lst.append(audio)
            label_lst.append(el["transcript"])
        # language_ids_lst.append(el["language_ids"])
        # lang_mix_id_lst.append(el["lang_mix_id"])

        selected_max_length = None
        for duration, length in zip(max_duration, max_length):
            if longest_audio <= duration:
                selected_max_length = length
                break

        try:
            batch = self.feature_extractor(audio_lst, sampling_rate=sr, return_tensors="pt")  # , padding=False)
        # print(batch)
        # print(batch.input_features.shape)
        # batch = self.processor.feature_extractor.pad(
        #                 batch,
        #                 padding="max_length",
        #                 max_length=selected_max_length,
        #                 return_tensors="pt",
        #                 )
        # print(batch)
        # print(batch.input_features.shape)
        except Exception as e:
            print("======")
            for el in features:
                print(el)
                audio = el["audio"]["array"]
                print(len(audio), flush=True)
            raise e

        if self.do_augment:
            input_features = batch.input_features
            input_features = self.spec_time_masking(input_features)
            input_features = self.spec_freq_masking(input_features)
            batch.input_features = input_features
        # print("==")
        # print(batch)
        #     #  print(batch["input_features"].shape)
        # labels_batch = self.processor.tokenizer(label_lst, src_lang=language, tgt_lang=language, return_tensors="pt", padding=True)
        labels_batch = self.text_processor(label_lst, return_tensors="pt", add_special_tokens=False, padding=True)

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		
        # 2) vectorized lookup: [B, L] of script‐labels
        script_ids = self.id2label[labels_batch.input_ids]             # [B, L]
		# 3) mask out pads/specials
        script_ids = script_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)
		# 4) shift off the first token to align with your labels[:,1:]
        batch["script_ids"] = script_ids[:, 1:]           # [B, L-1]
       
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        batch["decoder_input_ids"] = labels_batch.input_ids[:, :-1]
        batch["decoder_attention_mask"] = labels_batch.attention_mask[:, :-1]
        batch["labels"] = labels[:, 1:]

        # batch["lang_decoder_input_ids"] = lang_labels
        # batch["lang_labels"] = lang_labels
        # batch["only_lid_task"] = torch.Tensor([True])
        # batch["decoder_lang_mix_id"] = language_ids_lst

        return batch
