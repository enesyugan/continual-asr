import os
from datasets import Dataset, Audio, concatenate_datasets
from tqdm import tqdm
import re
import sys
import random
import numpy as np
import torch
from normalizers import ArNormalizer
# from trainer_mem import MemSeq2SeqTrainer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# TODO: change this
sys.path.append('/home/eugan/repos/yapay-net/src/hug/trainer/')


def normalize_text(utterance, language):
    arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')  # |\.
    cyrillic_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')
    japanese_filter = re.compile(
        r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.|\u3002|\u300C|\u300D|\uFF08|\uFF09|\uFF0C|\uFF1F|\uFF01|\uFF1A|\uFF1B')

    # english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:')#|\.
    if language == "ar":
        return re.subn(arabic_filter, '', utterance)[0].lower()
    elif language == "en" or language == "de" or language == "es" or language == "tr":
        return re.subn(english_filter, '', utterance)[0].lower()
    elif language == "uk":
        return re.subn(cyrillic_filter, '', utterance)[0].lower()
    elif language == "zh" or language == "ja":
        return re.subn(japanese_filter, '', utterance)[0].lower()
    else:
        raise ValueError(f'Text normalization for {language} is not supported')


CHARS_TO_IGNORE_PUNCT = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞", "؟",
                         "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "{", "}", "=",
                         "`", "_", "+", "<",
                         ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。", "、", "﹂", "﹁", "‧", "～", "﹏",
                         "，", "｛", "｝", "（",
                         "）", "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛",
                         "/", "\\", "º", "−",
                         "^", "ʻ", "ˆ", "]", "[", "-", "#"]

CHARS_TO_IGNORE = [
    "¿", "¡", ";", "；", ":", '"', "%", "�", "ʿ", "·", "჻", "~", "՞", "؟", "।", "॥", "«", "»", "„", "“", "”",
    "「", "」", "‘", "’", "《", "》", "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›",
    "©", "®", "—", "→", "﹂", "﹁", "‧", "～", "﹏", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
    "『", "』", "〝", "〟", "⟨", "⟩", "〜", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ", "]", "[", "-", "#"
]

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
chars_to_ignore_punct_re = f"[{re.escape(''.join(CHARS_TO_IGNORE_PUNCT))}]"


def remove_special_characters(text, lower, remove_punct):
    if remove_punct:
        text = re.sub(chars_to_ignore_punct_re, "", text)
    else:
        text = re.sub(chars_to_ignore_re, "", text)

    return text.lower() if lower else text


def file_exists(path):
    return os.path.exists(path)


mapper_orig = {
    "de": "<|de|>",
    "en": "<|en|>",
    "es": "<|es|>",
    "ar": "<|ar|>",
    "ua": "<|ua|>",
    "ja": "<|ja|>",
    "zh": "<|zh|>",
    "tr": "<|tr|>",
    "pa": "<|pa|>",
    "sq": "<|sq|>",
    "vi": "<|vi|>",
    "fa": "<|fa|>",
    "uz": "<|uz|>",
    "lv": "<|lv|>",
    "fi": "<|fi|>",
    "be": "<|be|>",
    "et": "<|et|>",
    "bn": "<|bn|>",
    "mix": "mix",
    "<unk>": "<unk>",
    "<eos>": "<eos>",
    "def": "def",
}

mapper_new = {
    "de": 2,
    "en": 0,
    "es": 3,
    "ar": 4,
    "ua": 5,
    "ja": 6,
    "zh": 1,
    "tr": 8,
    "mix": 0,
    "<unk>": 10,
    "<eos>": 11,
}


def detect_language(text, mapper):
    def detect_language_word(word, mapper):
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        mandarin_pattern = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')
        latin_pattern = re.compile(r'[a-zA-Z]')
        number_pattern = re.compile(r'^-?\d+(\.\d+)?$')

        if number_pattern.match(word):
            return mapper["en"]

        contains_arabic = arabic_pattern.search(word)
        contains_latin = latin_pattern.search(word)
        contain_mandarin = mandarin_pattern.search(word)

        if contain_mandarin and contains_latin and contains_arabic:
            print("WTF all 3")
            print(ASD)
        elif contains_arabic and contains_latin:
            print(ASD)
            return mapper['mix']
        elif contain_mandarin:
            return mapper["zh"]
        elif contains_arabic:
            print(AR)
            return mapper['ar']
        elif contains_latin:
            return mapper['en']
        else:
            print(f"word: {word}")
            # print(UNK)
            return mapper["en"]
            # return mapper["<unk>"]

    tmp = list()
    for word in text.split():
        tmp.append(detect_language_word(word, mapper))
    return tmp


def load_asr_dataset(file_path, language,
                     lower, remove_punct, special_char_removal=True):
    csw = True
    normalizers = []
    if language == "ar":
        normalizers.append(ArNormalizer(norm_unicode=False, norm_orthographic=False, remove_diacrits=True))

    if type(language) is not list:
        language = [language]
        csw = False

    # Read the content of the STM filea
    with open(file_path, 'r', encoding='utf-8') as stm_file:
        lines = stm_file.readlines()

    # Process lines to create a list of dictionaries
    data = []
    skipped = 0
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) < 6:
            # print(line)
            skipped += 1;
            continue
        try:
            uid, wavpath, start, end, duration, transcript = parts
        except Exception as e:
            uid, wavpath, start, end, duration = parts[:5]
            transcript = " ".join(parts[5:])

        if "/" not in wavpath:
            skipped += 1;
            continue
        # if not os.path.exists(wavpath): skipped +=1; continue

        duration = float(duration)  # ms
        if duration < 500 or duration > 20000: skipped += 1; continue

        if special_char_removal:
            transcript = remove_special_characters(transcript, lower, remove_punct)
        if transcript.strip() == "":
            skipped += 1
            continue
        if normalizers:
            for normalizer in normalizers:
                transcript = normalizer(transcript)

        transcript = " ".join(transcript.split())

        language_str = ",".join(language)

        if csw:
            # print(line)
            language_ids = detect_language(transcript, mapper_orig)
            lang_mix_id = -100
        else:
            # print(ENES)
            lang_mix_id = mapper_orig[language[0]]
            language_ids = [mapper_orig[language[0]] for _ in transcript.split()]

        # if len(parts) >= 3:  # Assuming at least 5 columns in STM file
        # transcript = "<|startoftranscript|><|{}|><|transcribe|><|notimestamps|> {}<|endoftext|>".format(language[0],transcript)
        transcript = "<|startoftranscript|><|{}|><|transcribe|><|notimestamps|> {}<|endoftext|>".format(language[0],
                                                                                                        transcript)

        data.append({
            # 'audio': audio.raw_data,
            # 'audio_path': parts[0],
            'uid': uid,
            'audio': wavpath,
            'transcript': transcript,
            'start': start,
            'end': end,
            'duration': duration,
            'language': language_str,
            'language_ids': language_ids,
            'lang_mix_id': lang_mix_id,
        })
        # print(language_ids)

    # Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)
    print(dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # print(dataset[0]["audio"])
    # print(type(dataset[0]["audio"]["array"]))
    # print(ASD)
    # for el in dataset:
    #     print(el)
    # print(AD)
    print("{}/{} skipped".format(skipped, len(lines)))
    return dataset


def find_index(x, ms, pseudo_csw_amount):
    check_range, counter = pseudo_csw_amount, 0
    while check_range < len(x) - 1:
        if x[check_range]["duration"] >= ms:
            break
        counter += 1
        check_range = pseudo_csw_amount * counter
    return min(check_range, len(x) - 1)


def do_csw_backup(train_list, pseudo_csw_amount, train=True):
    dummy_audio = None
    data_list = list()
    for x in train_list:
        x = x.sort("duration", reverse=False)
        if dummy_audio == None and x[0]["duration"] < 2000:
            dummy_path = x[0]["audio"]["path"]
        if train:
            ten_sec_idx = find_index(x, 10 * 1000, pseudo_csw_amount)
            if ten_sec_idx >= len(x) - 1 or int(pseudo_csw_amount) >= ten_sec_idx:
                tmp = x
            else:
                tmp_ten = x.train_test_split(test_size=int(ten_sec_idx), shuffle=False)["test"]
                tmp = tmp_ten.train_test_split(test_size=int(pseudo_csw_amount), shuffle=True)["test"]
            tmp = tmp.remove_columns(["audio"])
        else:
            tmp = x.train_test_split(test_size=int(pseudo_csw_amount), shuffle=False)
            tmp = tmp.remove_columns(["audio"])["test"]
        data_list.append(tmp)

    csw_data_list = list()
    number_of_elements = pseudo_csw_amount * 8
    used_idx = {}
    used_entry = False
    duration_20 = 0
    duration_30 = 0
    pbar = tqdm(total=number_of_elements)
    while number_of_elements > 0:
        if random.randint(0, 1) == 0:
            number_of_switches = 2
        else:
            number_of_switches = 3

        random_dataset = data_list[random.randint(0, len(data_list) - 1)]
        random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
        duration = random_entry["duration"]
        while duration > 10 * 1000:
            random_dataset = data_list[random.randint(0, len(data_list) - 1)]
            random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
            duration = random_entry["duration"]
        uid_list = [random_entry["uid"]]

        transcript = random_entry["transcript"].strip().split(" ", 1)[-1].replace("<|endoftext|>", "")

        start = random_entry["start"]
        end = random_entry["end"]

        language = ["<|{}|>".format(random_entry["language"])]
        number_of_elements -= 1
        pbar.update(1)

        for i in range(number_of_switches - 1):
            random_dataset = data_list[random.randint(0, len(data_list) - 1)]
            random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
            new_duration = random_entry["duration"]
            while (duration + new_duration > 15000 and i == 0) or (duration + new_duration > 20000 and i == 1):
                # print("{}+{}={}".format(duration, new_duration, duration+new_duration))
                random_dataset = data_list[random.randint(0, len(data_list) - 1)]
                random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
                new_duration = random_entry["duration"]

            duration += new_duration

            uid_list.append(random_entry["uid"])
            new_transcript = "{} {}".format(transcript, random_entry["transcript"].strip().split(" ", 1)[-1].replace(
                "<|endoftext|>", ""))
            transcript = new_transcript
            language.append("<|{}|>".format(random_entry["language"]))
            number_of_elements -= 1
            pbar.update(1)
            print(transcript)

        csw_transcript = "<|startoftranscript|>{}<|transcribe|><|notimestamps|> {}<|endoftext|>".format(
            "".join(language), transcript)

        entry = {
            "uid": "--".join(uid_list),
            "audio": dummy_path,
            "transcript": csw_transcript,
            "start": start,
            "end": end,
            "duration": duration,
            "language": "--".join(language),
        }
        csw_data_list.append(entry)
        if duration > 20000: duration_20 += 1
        if duration > 30000: duration_30 += 1

    csw_dataset = Dataset.from_list(csw_data_list)
    print(csw_dataset)
    csw_dataset = csw_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("EXcced: 20: {} 30: {}".format(duration_20, duration_30))
    pbar.close()
    return csw_dataset


def get_train_dev(config, special_char_removal=True):
    """
    Iterates over each dataset in 'config', prints dataset name,
    handles 'path'/'dev_path' as single string or list of strings,
    prints warnings if empty, and extracts other fields if present:
      - language
      - dev_split_size
      - lower
      - text_preprocessing
    """
    train_data_dict = {}
    dev_data_list = []

    # Extract the dictionary of datasets from the config
    datasets_dict = config.get("datasets", {})

    if not datasets_dict:
        print("No datasets found in config under 'datasets' key.")
        return

    for dataset_name, dataset_config in datasets_dict.items():
        print(f"\nProcessing dataset: '{dataset_name}'")

        # Extract fields with defaults
        language = dataset_config.get("language", None)
        dev_split_size = dataset_config.get("dev_split_size", 0)
        lower = dataset_config.get("lower", False)
        text_preprocessing = dataset_config.get("text_preprocessing", None)
        remove_punct = dataset_config.get("remove_punct", False)

        # Path can be string, list, or missing
        raw_path = dataset_config.get("path", "")
        if isinstance(raw_path, str):
            # Convert single string to list if non-empty
            path_list = [raw_path] if raw_path else []
        elif isinstance(raw_path, list):
            path_list = raw_path
        else:
            print(f"  WARNING: 'path' for '{dataset_name}' is neither string nor list. Found type: {type(raw_path)}")
            path_list = []

        # Dev path can be string, list, or missing
        raw_dev_path = dataset_config.get("dev_path", "")
        if isinstance(raw_dev_path, str):
            dev_path_list = [raw_dev_path] if raw_dev_path else []
        elif isinstance(raw_dev_path, list):
            dev_path_list = raw_dev_path
        else:
            print(
                f"  WARNING: 'dev_path' for '{dataset_name}' is neither string nor list. Found type: {type(raw_dev_path)}")
            dev_path_list = []

        # Check if we ended up with any valid paths
        if not path_list:
            print(f"  WARNING: No data path provided for '{dataset_name}'!")

        # Print the final extracted values
        print(f"  path               = {path_list}")
        print(f"  dev_path           = {dev_path_list}")
        print(f"  language           = {language}")
        print(f"  dev_split_size     = {dev_split_size}")
        print(f"  lower              = {lower}")
        print(f"  remove_punct       = {remove_punct}")
        print(f"  text_preprocessing = {text_preprocessing}")

        train_list = list()
        dev_list = list()

        for tr_path in path_list:
            train_data = load_asr_dataset(tr_path, language,
                                          lower, remove_punct,
                                          special_char_removal=special_char_removal).shuffle(seed=seed)
            if len(dev_path_list) == 0 and dev_split_size != 0:
                if isinstance(dev_split_size, float): dev_split_size = min(3000, int(dev_split_size * len(
                    train_data)))  # Use x% of the data or 3000 samples max
                tmp = train_data.train_test_split(test_size=dev_split_size)
                train_data = tmp["train"]
                dev_data = tmp["test"]
                dev_list.append(dev_data)

            train_list.append(train_data)

        if len(train_list) > 0:
            train_data_dict[dataset_name] = concatenate_datasets(train_list).shuffle(seed=seed)

        for dev_path in dev_path_list:
            dev_data = load_asr_dataset(dev_path, language,
                                        lower, remove_punct, special_char_removal=special_char_removal)
            if dev_split_size != 0:
                if isinstance(dev_split_size, float): dev_split_size = min(3000, int(dev_split_size * len(
                    train_data)))  # Use x% of the data or 3000 samples max
                if isinstance(dev_split_size, int): dev_split_size = min(dev_split_size, len(dev_data)-1)

                dev_data = dev_data.shuffle(seed=seed)
                tmp = dev_data.train_test_split(test_size=dev_split_size)
                dev_data = tmp["test"]

            dev_list.append(dev_data)

        if len(dev_list) > 0:
            dev_data_list.append(concatenate_datasets(dev_list))

    dev_data_concat = concatenate_datasets(dev_data_list)
    return train_data_dict, dev_data_concat
