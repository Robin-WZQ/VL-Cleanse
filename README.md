# 🛡️ VL-Cleanse: Assimilation-Guided Backdoor Identification and Mitigation for Vision-Language Pretrained Encoders

> Zhongqi Wang, Jie Zhang*, Shiguang Shan, Xilin Chen<br>
> *Corresponding author

VL-Cleanse is a unified framework for model-level backdoor identification, sample-level detection, and backdoor mitigation in vision-language pretrained (VLP) encoders. It recovers backdoor evidence with an assimilation-guided optimization objective and reuses the recovered feature, trigger embedding, and soft mask throughout the defense pipeline.

> **Release scope.** The main scripts in this directory currently implement the visual-encoder pipeline with Hugging Face `CLIPVisionModel`/`CLIPModel`. Utilities for additional experimental settings are provided under `Utils/`.

## 🔥 News

- **[2026/08/26]** We release the visual-encoder code for model identification, sample detection, and model mitigation.
- **[Coming soon]** Training data and evaluated backdoored model checkpoints will be released.

## 👀 Overview

VL-Cleanse follows a three-stage defense pipeline:

1. **Model-level identification.** Optimize a continuous image probe and a differentiable soft mask on benign auxiliary images. Backdoored encoders exhibit distinctive assimilation and convergence behavior.
2. **Sample-level detection.** Reuse the recovered backdoor feature as a lightweight signature and score input images by feature similarity.
3. **Model mitigation.** Reuse the recovered trigger embedding and mask to construct reversed samples, then fine-tune the compromised visual encoder with a small clean image-text set.

## 🧭 Getting Started

### Environment Requirement 🌍

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

1. Clone the repository and enter the code directory:

```bash
git clone https://github.com/Robin-WZQ/VL-Cleanse.git
cd VL-Cleanse
```

2. Create a virtual environment:

```bash
conda create -n vl-cleanse python=3.10 -y
conda activate vl-cleanse
python -m pip install --upgrade pip
```

3. Install PyTorch for your CUDA version by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the required packages:

```bash
pip install torch torchvision
pip install transformers numpy pillow tqdm matplotlib scikit-learn
```

The optional scripts in `Utils/` may additionally require:

```bash
pip install diffusers sentence-transformers
```

### Data Download ⬇️

#### Train_data

Download the benign auxiliary images used for feature inversion **[HERE](TRAIN_DATA_DOWNLOAD_URL_TO_BE_ADDED)** and place the extracted images directly under `Data/Train_data/`.

> **Download link placeholder:** replace `TRAIN_DATA_DOWNLOAD_URL_TO_BE_ADDED` with the public URL before release.

The training directory should contain image files directly, without class subdirectories:

```text
Data/
└── Train_data/
    ├── 000000.jpg
    ├── 000001.jpg
    ├── 000002.png
    └── ...
```

`model_identification.py` uses up to 5,000 images by default and performs an internal 80/20 train-test split.

#### Backdoored Models

Download the evaluated backdoored CLIP checkpoints **[HERE](BACKDOORED_MODEL_DOWNLOAD_URL_TO_BE_ADDED)** and extract them under `Models/Backdoor_models/`.

> **Download link placeholder:** replace `BACKDOORED_MODEL_DOWNLOAD_URL_TO_BE_ADDED` with the public URL before release.

Each checkpoint is expected to follow this layout:

```text
Models/
└── Backdoor_models/
    └── CLIP-L-336/
        ├── badnet/
        │   └── poisoned_model_3/
        │       └── pytorch_model.bin
        ├── badencoder/
        ├── PBCL/
        └── ...
```

The corresponding clean base model can be provided as a local Hugging Face model directory or as the model identifier [`openai/clip-vit-large-patch14-336`](https://huggingface.co/openai/clip-vit-large-patch14-336).

#### Sample-Detection Test Images

To reproduce sample-level ROC-AUC evaluation, prepare benign and triggered images as follows:

```text
Data/
└── Test_data/
    ├── Benign_CLIP/
    │   ├── benign_000.jpg
    │   └── ...
    └── BadNet/
        ├── backdoor_000.jpg
        └── ...
```

#### Mitigation Data

Model mitigation reads CC3M-style tar archives. Each archive must contain matched `.jpg` and `.txt` members with the same basename:

```text
Data/
└── CC3M/
    ├── cc3m-train-0000.tar
    ├── cc3m-train-0001.tar
    └── ...
```

## 🗂️ Project Structure

```text
VL-Cleanse/
├── model_identification.py       # model-level backdoor identification and evidence recovery
├── sample_detection.py           # sample-level detection and ROC-AUC evaluation
├── model_mitigation.py           # evidence-guided clean fine-tuning
├── test.py                       # identification metrics used during optimization
├── Utils/                        # datasets, losses, generation, and analysis utilities
├── Data/
│   ├── Train_data/
│   ├── Test_data/
│   └── CC3M/
├── Models/
│   └── Backdoor_models/
├── Results/                       # recovered evidence and identification logs
└── Mitigation/                    # mitigated checkpoints
```

## 🏃 Running Scripts

Run all commands from the repository root. The examples below use BadNet checkpoint `poisoned_model_3`; replace paths and the GPU index for your setting.

### Model-Level Backdoor Identification 🔎

```bash
python model_identification.py \
  --base_model_name openai/clip-vit-large-patch14-336 \
  --model_name ./Models/Backdoor_models/CLIP-L-336/badnet/poisoned_model_3/pytorch_model.bin \
  --dataset_path ./Data/Train_data \
  --output_folder ./Results/badnet/CLIP-L-336/poisoned_model_3 \
  --device cuda:0 \
  --image_size 336 \
  --scale 5000 \
  --epochs 1
```

Important options:

- `--lr`: learning rate for probe inversion; default `8e-2`.
- `--beta`: similarity-loss coefficient; default `2`.
- `--gamma`: soft-mask size coefficient; default `0.01`.
- `--threshold_same_target`: same-target decision threshold; default `0.949`.
- `--backdoored_model`: ground-truth label written to the log; it does not control the detector.

The script writes:

```text
Results/badnet/CLIP-L-336/poisoned_model_3/
├── Backdoor_Embedding.pt
├── Backdoor_Feature.pt
├── Soft_Mask.pt
├── params.pt
├── loss.txt
└── log.txt
```

`log.txt` reports the detection result, same-target ratio, final losses, recovered mask parameters, and runtime.

### Sample-Level Backdoor Detection 🧪

After model identification, use the recovered `Backdoor_Feature.pt` to evaluate benign and triggered samples:

```bash
python sample_detection.py \
  --base_model_name openai/clip-vit-large-patch14-336 \
  --model_name ./Models/Backdoor_models/CLIP-L-336/badnet/poisoned_model_3/pytorch_model.bin \
  --feature_path ./Results/badnet/CLIP-L-336/poisoned_model_3/Backdoor_Feature.pt \
  --benign_folder ./Data/Test_data/Benign_CLIP \
  --backdoor_folder ./Data/Test_data/BadNet \
  --output_path ./Results/badnet/CLIP-L-336/poisoned_model_3/sample_detection_roc.png \
  --device cuda:0 \
  --batch_size 64
```

The script prints the number of samples and ROC AUC, and saves the ROC curve to `--output_path`. Use `--limit N` for a quick evaluation on at most `N` images from each class.

### Backdoor Mitigation ⚒️

The mitigation stage uses the recovered `Soft_Mask.pt` and `Backdoor_Embedding.pt` to construct reversed samples and fine-tunes the visual encoder on clean image-text pairs:

```bash
python model_mitigation.py \
  --model_name openai/clip-vit-large-patch14-336 \
  --backdoor_model_root ./Models/Backdoor_models/CLIP-L-336 \
  --attack badnet \
  --cc3m_root ./Data/CC3M \
  --split train \
  --train_tars 1 \
  --max_samples 100 \
  --reversed_path ./Results \
  --save_path ./Mitigation/Post_training/VL-Cleanse/Models \
  --device cuda:0 \
  --batch_size 11 \
  --epochs 10 \
  --lr 4.5e-6
```

Supported `--attack` values are:

```text
BadVision, BadVision_untargeted, badencoder, PBCL, WB, DRUPE,
badnet, sig, blended, wanet, refool
```

The current mitigation script selects checkpoint IDs according to the following built-in mapping:

- `refool` → `poisoned_model_2`
- `PBCL` → `poisoned_model_3`
- `BadVision_untargeted` → `poisoned_model_8`
- all other supported attacks → `poisoned_model_3`

Make sure both the downloaded model and recovered evidence use the corresponding checkpoint ID. Mitigated weights are saved after every epoch under:

```text
Mitigation/Post_training/VL-Cleanse/Models/
└── <attack>/
    └── poisoned_model_<id>/
        ├── epoch_1/pytorch_model.bin
        ├── epoch_2/pytorch_model.bin
        └── ...
```

## ⚠️ Notes

- Checkpoint architecture and `--base_model_name` must match.
- Directory names and capitalization must match the selected attack name, especially on Linux.
- The default device values in some scripts reflect a multi-GPU research server. Pass `--device cuda:0` or another available device explicitly.
- `model_identification.py` currently expects image files directly inside `Train_data/`.
- Several scripts under `Utils/` retain experiment-specific default paths. Pass their command-line path arguments or update those defaults before use.

## 📄 Citation

If you find VL-Cleanse useful in your research, please consider citing it. The entry below is provisional and should be replaced by the official publication record when available:

```bibtex
@article{wang2026vlcleanse,
  title   = {VL-Cleanse: Assimilation-Guided Backdoor Identification and Mitigation for Vision-Language Pretrained Encoders},
  author  = {Wang, Zhongqi and Zhang, Jie and Shan, Shiguang and Chen, Xilin},
  year    = {2026}
}
```

🤝 Feel free to open an issue for questions or discussions.
