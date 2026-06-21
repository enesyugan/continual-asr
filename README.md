# Continual ASR: Multilingual and Code-Switching Speech Recognition

Research code for continual learning, multilingual automatic speech recognition (ASR), and code-switching speech recognition. This repository contains training and adaptation methods for Whisper and Qwen2-Audio, including LoRA, Bayesian low-rank factorization, batch ensembles, weight centralization, teacher distillation, and memory-efficient ASR fine-tuning.

## Research Topics

The code supports research on:

- Continual learning for ASR
- Multilingual ASR model adaptation
- Code-switching speech recognition
- Catastrophic forgetting in speech recognition
- Low-rank adaptation for speech models
- Bayesian LoRA / Bayesian low-rank factorization
- Whisper fine-tuning and Whisper continual learning
- Qwen2-Audio adaptation for code-switching ASR

---

## Papers

This repository contains code for the following papers:

### 1. Bayesian Low-Rank Factorization for Robust Model Adaptation

Implements Bayesian low-rank adaptation methods for robust ASR model adaptation, including Bayesian LoRA-style modules for Whisper.

### 2. Weight Factorization and Centralization for Continual Learning in Speech Recognition

Implements weight factorization and centralization methods for continual learning in automatic speech recognition.

### 3. Adding Robust Code-Switching Capabilities to High Performance Multilingual ASR

Provides training and adaptation code for improving multilingual ASR models on code-switching speech.

### 4. Adapting Language Balance in Code-Switching Speech

Code is available in the `mt-pier-focal-work` branch.

---

## Repository Structure

```text
whisper/           Whisper training, decoding, continual learning, LoRA, BNN-LoRA
qwen2/             Qwen2-Audio code-switching ASR adaptation
loras/             LoRA, sparse LoRA, and Bayesian low-rank modules
batch_ensembles/   Batch ensemble layers and Whisper variants
optimized/         Memory-efficient optimized layers
extensions/        CUDA/C++ extensions for efficient training
triton/            Triton kernels
```

---

## Main Features

- Fine-tuning OpenAI Whisper for multilingual ASR
- Continual learning experiments for ASR
- Code-switching ASR training with Whisper and Qwen2-Audio
- LoRA, PiSSA, OLoRA, EVA, RS-LoRA, and DoRA-style low-rank adaptation
- Bayesian LoRA / Bayesian low-rank factorization
- Weight centralization and factorization methods
- Batch ensemble adaptation for speech recognition
- Teacher distillation from pretrained Whisper
- SpecAugment, EMA, FSDP, FlashAttention, and memory-efficient training support

---

## Example Entry Points

### Continual Whisper Training

```bash
python whisper/train_whisper_continual.py \
    -dataset /path/to/dataset \
    -output outputs/
```

### Bayesian LoRA / Low-Rank Adaptation for Whisper

```bash
python whisper/train_bnnlora.py \
    -data_config config.yaml \
    -model_size large \
    -low_rank_type bayesian
```

### Code-Switching LoRA Training with Qwen2-Audio

```bash
python qwen2/train_csw_lora.py \
    -data_config config.yaml \
    -model_size large
```

---

## Models

This repository includes adaptation code for:

- OpenAI Whisper models
- `Qwen/Qwen2-Audio-7B`
- Hugging Face Transformers speech models

---

## Citation

If you use this repository, please cite the relevant paper:

```bibtex
@inproceedings{ugan2026bayesian,
  title={Bayesian Low-Rank Factorization for Robust Model Adaptation},
  author={Ugan, Enes Yavuz and Pham, Ngoc-Quan and Waibel, Alexander},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={18432--18436},
  year={2026},
  organization={IEEE}
}

@inproceedings{ugan2025weight,
  title={Weight Factorization and Centralization for Continual Learning in Speech Recognition},
  author={Ugan, Enes and Pham, Ngoc-Quan and Waibel, Alexander},
  booktitle={Proc. Interspeech 2025},
  pages={2200--2204},
  year={2025}
}

@article{ugan2025adapting,
  title={Adapting Language Balance in Code-Switching Speech},
  author={Ugan, Enes Yavuz and Pham, Ngoc-Quan and Waibel, Alexander},
  journal={arXiv preprint arXiv:2510.18724},
  year={2025}
}

Interspeech 2026: Adding Robust Code-Switching Capabilities to High Performance Multilingual ASR
```
