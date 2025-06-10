from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets, load_from_disk, \
    interleave_datasets
from datasets import Dataset
import torchaudio
import soundfile as sf  # fallback if needed
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask


import os
import sys
import re
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



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

from whisper.normalizers import ArNormalizer


def remove_special_characters(text, lower, remove_punct):
    if remove_punct:
        text = re.sub(chars_to_ignore_punct_re, "", text)
    else:
        text = re.sub(chars_to_ignore_re, "", text)

    return text.lower() if lower else text


def load_stm_file(stm_path,
                  language="en",
                  lower=False,
                  remove_punct=False,
                  special_char_removal=False):
    # List of dicts to feed into HuggingFace Dataset
    entries = []
    skipped = 0
    normalizers = []

    if language == "ar":
        normalizers.append(ArNormalizer(norm_unicode=False, norm_orthographic=False, remove_diacrits=True))

    with open(stm_path, "r", encoding="utf-8") as f:
        for line in f:
            _line = line.strip()
            if len(_line) == 0:
                skipped += 1
                continue

            parts = _line.split("\t")
            if len(parts) < 6:
                skipped += 1
                continue

            try:
                uid, wav_path, start, end, duration, transcript = parts
            except Exception as e:
                uid, wavpath, start, end, duration = parts[:5]
                transcript = " ".join(parts[5:])

            duration = float(duration)  # ms
            if duration < 500 or duration > 20000:
                skipped += 1
                continue

            if "/" not in wav_path:
                skipped += 1;
                continue

            if special_char_removal:
                transcript = remove_special_characters(transcript, lower, remove_punct)
            if transcript.strip() == "":
                skipped += 1
                continue
            if normalizers:
                for normalizer in normalizers:
                    transcript = normalizer(transcript)

            transcript = " ".join(transcript.split())

            entries.append({
                "id": uid,
                "wav_path": wav_path,
                "duration": duration,
                "text": transcript,
                "language": language
            })

    return Dataset.from_list(entries)


def get_train_dev(config,
                  special_char_removal=True,
                  seed=1234):
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
            train_data = load_stm_file(tr_path, language,
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
            dev_data = load_stm_file(dev_path, language,
                                        lower, remove_punct, special_char_removal=special_char_removal)

            if dev_split_size != 0:
                if isinstance(dev_split_size, float): dev_split_size = min(3000, int(dev_split_size * len(
                    train_data)))  # Use x% of the data or 3000 samples max
                dev_data = dev_data.shuffle(seed=seed)
                tmp = dev_data.train_test_split(test_size=dev_split_size)
                dev_data = tmp["test"]

            dev_list.append(dev_data)

        if len(dev_list) > 0:
            dev_data_list.append(concatenate_datasets(dev_list))

    dev_data_concat = concatenate_datasets(dev_data_list)
    return train_data_dict, dev_data_concat


if __name__ == "__main__":

    with open("fisher.yaml", "r") as f:
        data_config = yaml.safe_load(f)
        train_data_dict, dev_data = get_train_dev(data_config)

    train_data = concatenate_datasets(list(train_data_dict.values()))

    print(len(train_data))
    print(train_data[0])