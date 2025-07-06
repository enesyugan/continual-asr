import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask
import torchaudio.transforms as T
import torch

QWEN2_LANG_TO_TOKEN = {

    "en": "<|en|>",
    "es": "<|es|>",
    "zh": "<|zh|>",
    "ar": "<|ar|>",
    "de": "<|de|>",
    "tr": "<|tr|>",
    "ja": "<|ja|>",

    # TODO: automatically add from official repo
}

LANG_TOKENS = set(QWEN2_LANG_TO_TOKEN.values())


def load_audio(wav_path, augment=False, time_stretch=None,
               spec_time_masking=None,
               freq_time_masking=None):

    audio = torchaudio.load(wav_path)




class DataCollatorForQwen2:

    spec_time_masking = T.TimeMasking(time_mask_param=30)
    spec_freq_masking = T.FrequencyMasking(freq_mask_param=30)
    def __init__(self, processor,
                 prompt_template="<|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:",
                 eos_token="<|endoftext|>",
                 eos_token_id=151643,
                 audio_eos_token_id=151648,
                 augment=False,
                 include_text=False):
        self.processor = processor
        self.prompt_template = prompt_template
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        self.prompt_len = len(self.processor.tokenizer(self.prompt_template)["input_ids"])

        self.lang_ids = {token: processor.tokenizer.convert_tokens_to_ids(token) for token in LANG_TOKENS}

        self.do_augment = augment
        self.audio_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5, leave_length_unchanged=False),
            TimeMask(min_band_part=0.0, max_band_part=0.1, p=0.5),
        ])

        self.include_text = include_text

    def __call__(self, batch):
        # audio: waveform arrays
        # audio_list = [example["audio"]["array"] for example in batch]
        audio_list = [torchaudio.load(sample['wav_path'])[0].squeeze(0).numpy() for sample in batch]

        # Prepare prompt+target format for Qwen2-Audio
        text_list = [
            self._concat_prompt(example["text"], QWEN2_LANG_TO_TOKEN[example["language"]])
            for example in batch
        ]

        inputs = self.processor(
            audio=audio_list,
            text=text_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True  # pads both input_ids and audio
        )

        inputs_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs_ids.clone()

        batch_size = labels.size(0)

        for i in range(batch_size):

            label = labels[i]
            eos_pos = (label == self.audio_eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) == 0:
                raise ValueError("Missing <|audio_eos|> token.")
            eos_idx = eos_pos.item()

            # Mask everything up to and including <|audio_eos|>
            labels[i, :eos_idx + 1] = -100

            # Find the first <|lang|> token after <|audio_eos|>
            lang_id = self.lang_ids[QWEN2_LANG_TO_TOKEN[batch[i]["language"]]]
            lang_pos = (label == lang_id).nonzero(as_tuple=True)[0]
            if len(lang_pos) == 0:
                # print(batch[i]["language"], lang_id, lang_pos)
                raise ValueError("Missing <|lang|> token.")
            lang_idx = lang_pos.item()

            # Mask everything between eos and lang token (excluding lang token itself)
            if lang_idx > eos_idx + 1:
                labels[i, eos_idx + 1:lang_idx] = -100

        # labels.masked_fill_(attention_mask.ne(1), -100)
        inputs["labels"] = labels

        if self.do_augment:
            input_features = inputs["input_features"]
            input_features = self.spec_time_masking(input_features)
            input_features = self.spec_freq_masking(input_features)
            inputs["input_features"] = input_features
        # print(inputs.keys())

        if self.include_text:
            inputs["tgt_txt"] = [example["text"] for example in batch]

        return inputs

    def _concat_prompt(self, target_text, language_token):
        # add space between or not add space between?
        return f"{self.prompt_template} {language_token} {target_text} {self.eos_token}"


class EvalDataCollatorForQwen2:

    def __init__(self, processor,
                 prompt_template="<|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:",
                 eos_token="<|endoftext|>",
                 eos_token_id=151643,
                 audio_eos_token_id=151648):
        self.processor = processor
        self.prompt_template = prompt_template
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.audio_eos_token_id = audio_eos_token_id

    def __call__(self, batch):
        # audio: waveform arrays
        # audio_list = [example["audio"]["array"] for example in batch]
        audio_list = [torchaudio.load(sample['wav_path'])[0].squeeze(0).numpy() for sample in batch]

        # Prepare prompt+target format for Qwen2-Audio
        text_list = [
            example["text"]
            for example in batch
        ]

        data = {"audio": audio_list, "text": text_list}

        return data

    def _concat_prompt(self, target_text, language_token):
        # add space between or not add space between?
        return f"{self.prompt_template} {language_token} {target_text} {self.eos_token}"