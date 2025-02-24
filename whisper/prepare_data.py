import os
from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets, load_from_disk, \
    interleave_datasets
from pydub import AudioSegment
from tqdm import tqdm
import re
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, SeamlessM4TForSpeechToText, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
import sys
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask
import torchaudio.transforms as T
import copy
import numpy as np
from random import shuffle
from concurrent.futures import ThreadPoolExecutor
import torch

# from trainer_mem import MemSeq2SeqTrainer
from trainers.trainer_mem import MemSeq2SeqTrainer

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


# CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞", "؟",
# "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "{", "}", "=", "`", "_", "+", "<",
# ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。", "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（",
# "）", "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−",
# "^", "ʻ", "ˆ","]","[","-", "#"]

CHARS_TO_IGNORE = [
    "¿", "¡", ";", "；", ":", '"', "%", "�", "ʿ", "·", "჻", "~", "՞", "؟", "।", "॥", "«", "»", "„", "“", "”",
    "「", "」", "‘", "’", "《", "》", "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›",
    "©", "®", "—", "→", "﹂", "﹁", "‧", "～", "﹏", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
    "『", "』", "〝", "〟", "⟨", "⟩", "〜", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ", "]", "[", "-", "#"
]

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters(text, lower):
    if chars_to_ignore_re is not None:
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


def load_asr_dataset(file_path, language, lower):
    csw = True
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

        transcript = remove_special_characters(transcript, lower)
        if transcript.strip() == "":
            skipped += 1
            continue

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


#
# def get_data(debug=False):
#     datasets = {}
#
#     # Iterate over all directories in the current working directory
#     for dir_name in os.listdir('../../../../repos/tmp'):
#         print(f"Loading {dir_name} ...")
#         # Check if it's a directory and ends with "..data"
#         if os.path.isdir(dir_name) and dir_name.endswith(".data"):
#             # Define the two files we need to load
#             all_tr_file = os.path.join(dir_name, "all-tr.stm")
#             small_dev_file = os.path.join(dir_name, "small-dev.stm")
#
#             # Check if the files exist in the directory
#             if os.path.isfile(all_tr_file) and os.path.isfile(small_dev_file):
#                 # Load the datasets using the hypothetical load_asr_dataset function
#                 all_tr_dataset = load_asr_dataset(all_tr_file, "def").shuffle(seed=42)
#                 small_dev_dataset = load_asr_dataset(small_dev_file, "def")
#
#                 # Use directory name and file name to create keys for the dictionary
#                 datasets[f"{dir_name}_all-tr"] = all_tr_dataset
#                 datasets[f"{dir_name}_small-dev"] = small_dev_dataset
#
#     dev_list = [v for k, v in datasets.items() if "_small-dev" in k]
#     datasets = {k: v for k, v in datasets.items() if "_small-dev" not in k}
#     #    dev_list = list()
#     #    for k in datasets.keys():
#     #        if "_small-dev" in k:
#     #            dev_list.append(datasets[k])
#     #            del datasets[k]
#
#     #  datasets = {}
#
#     #  print("Loading ARZEN...")
#     #  arzen_train="/project/asr_systems/LT2022/codeswitching/data/ARX-ENG/arzen/ArzEn_SpeechCorpus_1.0/train-clip-ntags.stm"
#     #  arzen_dev = "/project/asr_systems/LT2022/codeswitching/data/ARX-ENG/arzen/ArzEn_SpeechCorpus_1.0/dev-clip-ntags.stm"
#     #  shuffle_arzen_train_dataset = load_asr_dataset(arzen_train, "ar").shuffle(seed=42)
#     #  arzen_train_dataset = load_asr_dataset(arzen_train, "ar")
#     #  arzen_dev_dataset = load_asr_dataset(arzen_dev, "ar")
#
#     print("Loading SEAME...")
#     seame_train = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/seame/train_clip.stm"
#     seame_train_dataset = load_asr_dataset(seame_train, "def")
#     tmp = seame_train_dataset.train_test_split(test_size=3000)
#     seame_train_dataset = tmp["train"]
#     seame_dev_dataset = tmp["test"]
#     shuffle_seame_train_dataset = seame_train_dataset.shuffle(seed=42)
#
#     dev_list.append(seame_dev_dataset)
#     #  print("Loadin ASCEND ...")
#     #  ascend_train="/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/ASCEND/train_notags.stm"
#     #  ascend_dev="/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/ASCEND/dev_notags.stm"
#     #  ascend_train_dataset = load_asr_dataset(ascend_train, "def").shuffle(seed=42)
#     #  ascend_dev_dataset = load_asr_dataset(ascend_dev, "def")
#
#     print("Loading Fisher...")
#     fisher_train = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_cs_train-transcript.stm"
#     fisher_dev = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_cs_dev-transcript.stm"
#     shuffle_fisher_train_dataset = load_asr_dataset(fisher_train, "es").shuffle(seed=42)
#     fisher_train_dataset = load_asr_dataset(fisher_train, "es")
#     fisher_dev_dataset = load_asr_dataset(fisher_dev, "es")
#
#     fisher_mono_train = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_mono-transcript.stm"
#     shuffle_fisher_mono_train_dataset = load_asr_dataset(fisher_mono_train, "es").shuffle(seed=42)
#
#     fisher_all_train_dataset = concatenate_datasets([shuffle_fisher_train_dataset, shuffle_fisher_mono_train_dataset])
#
#     #  print("Loading TALCS ...")
#     #  talcs_train = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/TALCS_corpus/train_set/train_time.stm"
#     #  talcs_dev = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/TALCS_corpus/dev_set/dev_time.stm"
#     #  talcs_train_dataset = load_asr_dataset(talcs_train, "def").shuffle(seed=42)
#     #  talcs_dev_dataset = load_asr_dataset(talcs_dev, "def")
#
#     # csw_train_dataset = concatenate_datasets([shuffle_arzen_train_dataset, shuffle_seame_train_dataset, ascend_train_dataset, shuffle_fisher_train_dataset])
#     # csw_dev_dataset = concatenate_datasets([arzen_dev_dataset, seame_dev_dataset, ascend_dev_dataset, fisher_dev_dataset])
#
#     datasets["csw_train_dataset"] = fisher_all_train_dataset  #csw_train_dataset
#     datasets["shuffle_seame_train_dataset"] = shuffle_seame_train_dataset
#     #datasets["talcs_train_dataset"] = talcs_train_dataset
#
#     # dev_list.append(csw_dev_dataset)
#     #dev_list.append(talcs_dev_dataset)
#     dev_list.append(fisher_dev_dataset)
#     all_dev_dataset = concatenate_datasets(dev_list)
#
#     print(datasets)
#     print("===" * 20)
#     print(dev_list)
#
#     print("=====")
#     print(f"Dev data: {all_dev_dataset}")
#
#     return datasets, all_dev_dataset, fisher_dev_dataset


def get_data_seame(lower, debug=False):
    datasets = {}

    dev_list = list()

    print("Loading SEAME...")
    seame_train = "seame_train_clip.stm"
    seame_train_dataset = load_asr_dataset(seame_train, "zh", lower)
    tmp = seame_train_dataset.train_test_split(test_size=3000)
    seame_train_dataset = tmp["train"]
    seame_dev_dataset = tmp["test"]
    shuffle_seame_train_dataset = seame_train_dataset.shuffle(seed=181195)

    datasets["shuffle_seame_train_dataset"] = shuffle_seame_train_dataset

    dev_list.append(seame_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_fisher(lower, debug=False):
    '''
    Spanish-English Telephone conversations
    '''
    datasets = {}

    dev_list = []

    print("Loading Fisher...")
    fisher_train = "fisher_train_cs_train-transcript.stm"
    fisher_dev = "fisher_train_cs_dev-transcript.stm"
    shuffle_fisher_train_dataset = load_asr_dataset(fisher_train, "es", lower).shuffle(seed=42)
    fisher_train_dataset = load_asr_dataset(fisher_train, "es", lower)
    fisher_dev_dataset = load_asr_dataset(fisher_dev, "es", lower)

    fisher_mono_train = "fisher_train_mono-transcript.stm"
    shuffle_fisher_mono_train_dataset = load_asr_dataset(fisher_mono_train, "es", lower).shuffle(seed=42)

    fisher_all_train_dataset = concatenate_datasets([shuffle_fisher_train_dataset, shuffle_fisher_mono_train_dataset])

    datasets["csw_train_dataset"] = fisher_all_train_dataset  # csw_train_dataset

    dev_list.append(fisher_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_arzen(lower, debug=False):
    '''
    Arabic-English: spontaneous conversational speech corpus informal interviews, Egyptian-English; Setting: all recordings were carried out in a soundproof room.
    '''
    datasets = {}

    dev_list = []

    print("Loading ARZEN...")
    arzen_train = "arzen_train-clip-ntags.stm"
    arzen_dev = "arzen_dev-clip-ntags.stm"

    shuffle_arzen_train_dataset = load_asr_dataset(arzen_train, "ar", lower).shuffle(seed=42)
    arzen_train_dataset = load_asr_dataset(arzen_train, "ar", lower)
    arzen_dev_dataset = load_asr_dataset(arzen_dev, "ar", lower)

    datasets["csw_train_dataset"] = shuffle_arzen_train_dataset
    dev_list.append(arzen_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_ascend(lower, debug=False):
    '''
  A Spontanenous Chinese-English from Hong Kong conversations about different topics; Setting: The recordings are made in a quiet classroom.
  Both speakers are seated across one another at a distance of ∼1 meter. Each speaker is equipped with a RODE SmartLav+ clip microphone as the recording device.
  The microphone is mounted on the speaker’s shirt collar.
  '''
    datasets = {}
    dev_list = []

    print("Loading ASCEND ...")
    ascend_train = "ascend_train_notags.stm"
    ascend_dev = "ascend_dev_notags.stm"

    shuffle_ascend_train_dataset = load_asr_dataset(ascend_train, "zh", lower).shuffle(seed=42)
    ascend_train_dataset = load_asr_dataset(ascend_train, "zh", lower)
    ascend_dev_dataset = load_asr_dataset(ascend_dev, "zh", lower)

    datasets["csw_train_dataset"] = shuffle_ascend_train_dataset
    dev_list.append(ascend_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_talcs(lower, debug=False):
    '''
  Spontaneous; Real online one-to-one English teaching Mandarin-English only teachers; different regions of China; Setting: recorded by the personal computer microphone
  '''
    datasets = {}
    dev_list = []

    print("Loading TALCS ...")
    talcs_train = "talcs_train_time.stm"
    talcs_dev = "talcs_dev_time.stm"

    shuffle_talcs_train_dataset = load_asr_dataset(talcs_train, "zh", lower).shuffle(seed=42)
    talcs_train_dataset = load_asr_dataset(talcs_train, "zh", lower)
    talcs_dev_dataset = load_asr_dataset(talcs_dev, "zh", lower)

    datasets["csw_train_dataset"] = shuffle_talcs_train_dataset
    dev_list.append(talcs_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_tunswitch(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading TunSwitch ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    tunswitch_csw_train = "tunswitch_csw_train.stm"
    tunswitch_csw_dev = "tunswitch_csw_dev.stm"

    
    fisher_dev = "fisher_train_cs_dev-transcript.stm"
    shuffle_tunswitch_csw_train_dataset = load_asr_dataset(tunswitch_csw_train, "ar", lower).shuffle(seed=42)
    #fisher_train_dataset = load_asr_dataset(tunswitch_csw_train, "ar")
    tunswitch_csw_dev_dataset = load_asr_dataset(tunswitch_csw_dev, "ar", lower)
  #  shuffle_tunswitch_mono_train_dataset = load_asr_dataset(tunswitch_mono_train, "ar").shuffle(seed=42)
  #  tunswitch_mono_dev_dataset = load_asr_dataset(tunswitch_mono_dev, "ar")


    #tunswitch_all_train_dataset = concatenate_datasets([shuffle_tunswitch_csw_train_dataset, shuffle_tunswitch_mono_train_dataset])

    datasets["csw_train_dataset"] = shuffle_tunswitch_csw_train_dataset  #csw_train_dataset

    dev_list.append(tunswitch_csw_dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_pa(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading PA (Punjabi) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    pa_train = "pa_train.mp3.stm"
    pa_dev = "pa_dev.mp3.stm"

    
    shuffle_pa_train_dataset = load_asr_dataset(pa_train, "pa", lower).shuffle(seed=42)
    #fisher_train_dataset = load_asr_dataset(tunswitch_csw_train, "ar")
    pa_dev_dataset = load_asr_dataset(pa_dev, "pa", lower)
  #  shuffle_tunswitch_mono_train_dataset = load_asr_dataset(tunswitch_mono_train, "ar").shuffle(seed=42)
  #  tunswitch_mono_dev_dataset = load_asr_dataset(tunswitch_mono_dev, "ar")


    #tunswitch_all_train_dataset = concatenate_datasets([shuffle_tunswitch_csw_train_dataset, shuffle_tunswitch_mono_train_dataset])

    datasets["csw_train_dataset"] = shuffle_pa_train_dataset  #csw_train_dataset

    dev_list.append(pa_dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset
    
def get_data_sq(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading SQ (Albanian) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    sq_train = "sq_train.mp3.stm"
    sq_dev = "sq_dev.mp3.stm"

    
    shuffle_sq_train_dataset = load_asr_dataset(sq_train, "sq", lower).shuffle(seed=42)
    sq_dev_dataset = load_asr_dataset(sq_dev, "sq", lower)


    #tunswitch_all_train_dataset = concatenate_datasets([shuffle_tunswitch_csw_train_dataset, shuffle_tunswitch_mono_train_dataset])

    datasets["csw_train_dataset"] = shuffle_sq_train_dataset  #csw_train_dataset

    dev_list.append(sq_dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_vi(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading VI (Vietnamese) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    vi_train = "vi_train.mp3.stm"
    vi_dev = "vi_dev.mp3.stm"

    
    shuffle_vi_train_dataset = load_asr_dataset(vi_train, "vi", lower).shuffle(seed=42)
    vi_dev_dataset = load_asr_dataset(vi_dev, "vi", lower)


    #tunswitch_all_train_dataset = concatenate_datasets([shuffle_tunswitch_csw_train_dataset, shuffle_tunswitch_mono_train_dataset])

    datasets["csw_train_dataset"] = shuffle_vi_train_dataset  #csw_train_dataset

    dev_list.append(vi_dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_tr(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading TR (Turkish) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "tr_train.mp3.stm"
    dev = "tr_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "tr", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "tr", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_ar(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading AR (Arabic) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "ar_train.mp3.stm"
    dev = "ar_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "ar", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "ar", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_fa(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading FA (Farsi/Persian) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "fa_train.mp3.stm"
    dev = "fa_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "fa", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "fa", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_uz(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading UZ (Uzbek) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "uz_train.mp3.stm"
    dev = "uz_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "uz", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "uz", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_lv(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading LV (Latvian) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "lv_train.mp3.stm"
    dev = "lv_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "lv", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "lv", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_fi(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading FI (Finish) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "fi_train.mp3.stm"
    dev = "fi_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "fi", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "fi", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_be(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading BE (Belarusian) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "be_train.mp3.stm"
    dev = "be_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "be", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "be", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset

def get_data_et(lower, debug=False):
    datasets = {}

    dev_list = []

    print("Loading ET (Estonian) ...")
   # tunswitch_mono_train = "tunswitch_mono_train.stm"
   # tunswitch_mono_dev = "tunswitch_mono_dev.stm"
    train = "et_train.mp3.stm"
    dev = "et_dev.mp3.stm"

    
    shuffle_train_dataset = load_asr_dataset(train, "et", lower).shuffle(seed=42)
    dev_dataset = load_asr_dataset(dev, "et", lower)

    datasets["csw_train_dataset"] = shuffle_train_dataset  #csw_train_dataset

    dev_list.append(dev_dataset)
   # dev_list.append(tunswitch_mono_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


flex_datasets = {
    "ar_cv": {"train": ["ar_cv_train_length.stm"], "dev": ["ar_cv_dev_length.stm"]}, #common voice
    "ar_mgb": {"train": ["ar_mgb_mgb-clip.stm"], "dev": None}, #mgb aljazeera
    "ar_ldclv": {"train": ["ar_ldclv_train_split_clip.stm"], "dev": None}, #ldc2006S29 Levantine arabic
    "ar_mediaspeech": {"train":  ["ar_mediaspeech_tr.mediaspeech.stm"], "dev": None}, # mediaspeech
    "ar_juansy": {"train": ["ar_juansy_juan-syrian-clip.stm"], "dev": None}, #levantine by juan
    "ar_mini": {"train": ["ar_mini_mini-clip.stm"], "dev": None}, #mini questionaire
    "ar_quran": {"train": ["ar_quran_train.not.stm"], "dev": None}, #Quran arabic
    "ar_tunmsa": {"train": ["ar_tunmsa_train.stm"], "dev": ["ar_tunmsa_dev.stm"]}, #tunisian msa
    "ar_tunmsaslt": {"train": ["ar_tunmsaslt_train.asr.stm"], "dev": ["ar_tunmsaslt_dev.asr.stm"]}, # tunisian msa slt ldc2022e01
    }


def get_data_flex(dataset, lower, debug=False):    
    datasets = {}

    dev_list = []
    
    lang = dataset.split("_")[0]
    # Get dataset paths from the dictionary
    dataset_info = flex_datasets.get(dataset, {})
    
    # Process train datasets
    train_datasets = []
    if "train" in dataset_info and dataset_info["train"]:
        train_files = dataset_info["train"]
        for train_file in train_files:
            train_datasets.append(load_asr_dataset(train_file, lang, lower))
    
    # Concatenate multiple train datasets if needed
    train_data = None
    if train_datasets:
        train_data = train_datasets[0] if len(train_datasets) == 1 else concatenate_datasets(train_datasets)
        train_data = train_data.shuffle(seed=42)  # Shuffle train dataset

    # Process dev datasets
    dev_datasets = []
    if "dev" in dataset_info and dataset_info["dev"]:
        dev_files = dataset_info["dev"]
        for dev_file in dev_files:
            dev_datasets.append(load_asr_dataset(dev_file, lang, lower))

        
    dev_data = None
    if dev_datasets:
        dev_data = dev_datasets[0] if len(dev_datasets) == 1 else concatenate_datasets(dev_datasets)
    # If no dev dataset exists, create one from the train dataset
    if dev_data is None and train_data is not None:
        total_samples = len(train_data)
        split_size = min(3000, int(0.1 * total_samples))  # Use 10% of the data or 3000 samples max
        if split_size > 0:
            tmp = train_data.train_test_split(test_size=split_size)
            train_data = tmp["train"]
            dev_data = tmp["test"]
        else:
            print(f"Warning: Not enough data to create a dev split for {dataset}. Using all data for training.")

    datasets[f"train_{dataset}"] = train_data  #csw_train_dataset
    return datasets, dev_data

    
    
