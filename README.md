# VLM Healthcare Bias Audit: CLIP & OpenCLIP

Repository for profession-wise bias evaluation of vision–language models (VLMs) in healthcare using FairFace.  
Implements streaming preprocessing, embedding extraction, and top-K retrieval analysis for four models:  

- **OpenAI CLIP**: ViT-B/16, ViT-B/32  
- **OpenCLIP**: ViT-L/14, ViT-H/14  

---

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Environment Setup](#environment-setup)
- [Model Scripts](#model-scripts)
  - [OpenAI CLIP (ViT-B/16)](./openai_clip_(vit_b_16).py)
  - [OpenAI CLIP (ViT-B/32)](./openai_clip_(vit_b_32).py)
  - [OpenCLIP (ViT-L/14)](./openai_clip_(vit_l_14).py)
  - [OpenCLIP (ViT-H/14)](./open_clip_(vit_h_14).py)
- [Code Instructions](#code-instructions)
- [Typical Workflow](#typical-workflow)
- [Outputs](#outputs)
- [Reproducibility Tips](#reproducibility-tips)
---

## Overview
This repository evaluates demographic bias for healthcare professions by probing VLMs with neutral, profession-aware prompts (e.g., "Photo of a cardiologist").  
It measures the demographic composition among the top-K retrieved faces from FairFace and optionally supports downstream bias scoring.

Each model script has two phases:

1. **Preprocessing & Embeddings** – Builds normalized image embeddings and metadata from FairFace.  
2. **Retrieval & Ranking** – Encodes a text prompt, ranks images by cosine similarity, and saves top-K matches with metadata to Excel.

---

## Folder Structure
- `openai_clip_(vit_b_16).py` — OpenAI CLIP ViT-B/16 pipeline (embeddings + retrieval)  
- `openai_clip_(vit_b_32).py` — OpenAI CLIP ViT-B/32 pipeline (embeddings + retrieval)  
- `open_clip_(vit_l_14).py` — OpenCLIP ViT-L/14 pipeline (embeddings + retrieval)  
- `open_clip_(vit_h_14).py` — OpenCLIP ViT-H/14 pipeline (embeddings + retrieval)  

> ⚠️ Each script defines its own `OUTPUT_DIR` and file paths. Adjust them if running outside Colab.

---

## Dependencies
Python 3.9+ recommended. Install packages via pip:

```bash
pip install transformers datasets torch torchvision torchaudio
pip install numpy pandas h5py pillow openpyxl tqdm
Optional:
accelerate or bitsandbytes for GPU memory optimization


# Install core dependencies
pip install transformers datasets torch torchvision torchaudio
pip install numpy pandas h5py pillow openpyxl tqdm

# Optional (for GPU memory optimization)
pip install accelerate bitsandbytes
```

---

## Datasets
- **[FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace)** via HuggingFace Datasets  
- Subsets: `1.25` and `0.25`  
- Splits are combined to expand coverage  
- Metadata includes:  
  - subset  
  - split  
  - index  
  - file path  
  - age  
  - gender  
  - race  


# Environment Setup
## Use a GPU runtime (Colab, Kaggle, or local CUDA machine).
- Recommended GPU: A100 or equivalent.

### Create and activate virtual environment
```bash
python -m venv .venv
```

### On Linux / macOS
```bash
source .venv/bin/activate
```

### On Windows
```bash
.venv\Scripts\activate
```

### Upgrade pip
```bash
pip install --upgrade pip
```


# Model Scripts

Each script performs **embedding creation** and **top-K retrieval**.  
Adjust `OUTPUT_DIR`, `embeddings_path`, `metadata_path`, `DEVICE`, `TOP_K`, and `PROFESSION` as needed.

---

## OpenAI CLIP (ViT-B/16)

- **Model**: `openai/clip-vit-base-patch16`  
- **OUTPUT_DIR**: `output_openai_clip`  
- **Run**:
```bash
python openai_clip_(vit_b_16).py
```

### Outputs:
- fairface_clip_embeddings.h5
- fairface_metadata.csv
- <Profession>.xlsx — top-K results


## OpenAI CLIP (ViT-B/32)

- **Model**: openai/clip-vit-base-patch32
- **OUTPUT_DIR**: output_openai_clip_vitb32
- **Run**:
```bash
python openai_clip_(vit_b_32).py
```

## OpenCLIP (ViT-L/14)

- **Model**: laion/CLIP-ViT-L-14
- **OUTPUT_DIR**: output_openclip_vitl14
- **Run**:
```bash
python open_clip_(vit_l_14).py
```

## OpenCLIP (ViT-H/14)

- **Model**: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- **OUTPUT_DIR**: output_openclip_vith14
- **Run**:
```bash
python open_clip_(vit_h_14).py
```

# Code Instructions

Each code file has **two main parts**:

1. **Part 1 – Embedding & Metadata Extraction**  
   - Streams the full FairFace dataset.  
   - Extracts and saves normalized image embeddings (`.h5`) and metadata (`.csv`).  
   - This step is **one-time only**: once embeddings are created, they can be reused.

2. **Part 2 – Retrieval & Ranking**  
   - Loads the precomputed embeddings (`.h5`) and metadata (`.csv`) from Part 1.  
   - Encodes a profession-specific text prompt (e.g., *“Photo of a cardiologist”*).  
   - Computes cosine similarity between the text and image embeddings.  
   - Selects the **top-100 images** based on similarity scores.  
   - Saves results (including demographics and similarity) into an Excel file (`.xlsx`) for further analysis.  

---

⚠️ **Important Note**  
- You do **not** need to rerun **Part 1** every time.  
- Run it once to build the embeddings.  
- For new professions, simply update the `PROFESSION` in **Part 2** and rerun only that section to generate new results.



# Typical Workflow

- Pick a model script (e.g., ViT-B/16) and set PROFESSION and output paths.
- Run the script to stream FairFace, compute normalized embeddings, and save metadata.
- Encode the text prompt, compute cosine similarity with image embeddings, select TOP_K, and export to Excel.
- Repeat for other models to compare distributions or compute cross-model metrics.


## Outputs

- **fairface_clip_embeddings.h5** — normalized CLIP image features
- **fairface_metadata.csv** — per-image metadata (age, gender, race)
- **<Profession>.xlsx** — top-K ranked results with similarity and demographics



# Reproducibility Tips

- GPU memory varies; reduce batch size for large models to avoid OOM
- Keep **embeddings_path** and **metadata_path** consistent to reuse features
- Normalize both image and text embeddings before similarity (handled in scripts)
- Pin package versions using requirements.txt for consistent results
