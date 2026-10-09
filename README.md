# GeoVP

<div align="center">

### A Unified Visual Prompting Framework for Multi-Source Remote Sensing Image Understanding

**Le Yu · Yuanwen Wang · Xiaotong Qi**

**Point · Box · Free-form** &nbsp; | &nbsp; **Optical · SAR · Infrared**

[Overview](#overview) · [Architecture](#architecture) · [GeoVP-650K](#geovp-650k) · [Results](#results) · [Getting Started](#getting-started) · [Citation](#citation)

</div>

GeoVP combines visual prompts with language instructions to understand user-selected regions in remote sensing imagery. A single framework connects global scene context, local region representations, and language reasoning across optical, synthetic aperture radar (SAR), and infrared images.

**Release scope:** This repository presents the GeoVP research framework and a partial code snapshot. The paper-specific Region-aware Encoder, complete visual prompt input pipeline, trained GeoVP weights, and GeoVP-650K annotations are not included in this release. The results below are reported from the manuscript; this snapshot alone does not reproduce them.

![Overview of GeoVP: task formulation, overall framework, and GeoVP-650K construction pipeline.](assets/readme/geovp-overview.png)

*Fig. 1. Overview of GeoVP for multi-source remote sensing visual prompting.*

## Overview

Text alone can be ambiguous when referring to small objects or irregular regions in complex remote sensing scenes. GeoVP provides an explicit spatial interface: select a point, draw a box, or specify a free-form region, then ask the model about the selected content.

- **Flexible visual prompting.** Points, bounding boxes, scribbles, polygons, and masks share a unified region representation.
- **Multi-source understanding.** Optical, SAR, and infrared imagery are organized within the same image–region–language framework.
- **Multiple semantic granularities.** Tasks include scene classification, referring object category recognition, region captioning, detailed description, and cross-region relationship analysis.
- **Cross-domain training.** A one-stage strategy aligns visual prompts with language reasoning across natural-image and remote sensing domains.

Example instructions for the paper's visual prompting formulation:

| Task | Instruction |
| --- | --- |
| Targeted recognition | What is the category of the object at the marked point? |
| Region captioning | Provide a concise caption for region 1. |
| Detailed description | Describe the objects and spatial layout inside the selected region. |
| Relationship analysis | How is region 1 spatially related to region 2? |
| Scene understanding | Summarize the overall scene and its main features. |

## Architecture

![Fig. 2. Overall architecture of GeoVP, showing multiscale visual feature encoding, region prompt encoding, text instruction encoding, alignment, language reasoning, and auxiliary pixel-level localization.](assets/readme/geovp-architecture.png)

*Fig. 2. Overall architecture of GeoVP.*

GeoVP takes an image, visual prompts, and a textual instruction as inputs:

1. **Hybrid Vision Encoder.** CLIP-ConvNeXt-L and DINOv2 ViT-L/14 extract complementary semantic and structural features. A 448 × 448 image is represented by one global view and four local views, each at 224 × 224 resolution.
2. **Region-aware Encoder.** Heterogeneous prompts are converted into foreground masks. The encoder samples 512 foreground points and applies two farthest-point-sampling and nearest-neighbor aggregation blocks to produce compact region representations.
3. **Alignment Projection Module.** Lightweight projectors map image features and region features into the language model's embedding space. Region tokens combine prompt-conditioned structure with local visual semantics.
4. **Language reasoning.** LLaMA 2-13B jointly processes image-level tokens, region-level tokens, and text instructions to generate a response.
5. **Pixel-Level Localization Module.** An auxiliary SAM-based branch supplies qualitative mask visualizations for prompted regions. These masks are localization cues; the paper does not report quantitative segmentation performance.

During one-stage training, the visual backbones and SAM components remain frozen. The Region-aware Encoder, alignment projectors, and language-side LoRA parameters are optimized with an autoregressive language modeling objective.

## GeoVP-650K

GeoVP-650K contains **654.45K image–prompt–text triplets**, covering multiple sensing modalities and task granularities. The count refers to reconstructed training triplets; multiple triplets may share the same source image.

| Data branch | Supervision |
| --- | --- |
| Optical | Point-based object recognition, box-based region captioning, image captioning, and scene classification |
| SAR | Box-based object recognition, regional descriptions, and relationship analysis |
| Infrared | Box-based object recognition, regional descriptions, and relationship analysis |
| GPT-4V-assisted annotations | Fine-grained captions and relations with region-reference checks, consistency filtering, and manual inspection of a stratified subset |

Point, box, and free-form annotations are standardized into a shared prompting format. Free-form prompts derived from point or box annotations are not counted as additional triplets.

## Results

The following headline results are reported in the GeoVP manuscript:

| Benchmark | Evaluation setting | Metric | GeoVP |
| --- | --- | --- | ---: |
| AID | Zero-shot scene classification | Average accuracy | **79.05%** |
| UCMerced | Cross-task scene classification | Average accuracy | **87.55%** |
| DIOR-RSVG | Held-out, box-prompt-conditioned region understanding | Average semantic IoU (S-IoU) | **98.25%** |

AID is excluded from GeoVP-650K. UCMerced is evaluated across tasks because UCM-Captions shares its image source with the classification benchmark. DIOR-RSVG uses its held-out split. S-IoU measures semantic category agreement in this evaluation and should not be interpreted as mask segmentation IoU.

### Training configuration in the paper

| Setting | Value |
| --- | --- |
| Language backbone | LLaMA 2-13B |
| Visual backbones | CLIP-ConvNeXt-L + DINOv2 ViT-L/14 |
| Input resolution | 448 × 448 |
| Optimizer | AdamW |
| Learning rate / weight decay | 2 × 10⁻⁵ / 0.01 |
| Schedule | Cosine decay; 0.03 warm-up ratio |
| LoRA | Rank 16; scaling factor 32; upper 30 Transformer layers |
| Training length | One epoch |
| Effective batch size | 64 |
| Hardware | 8 NVIDIA A100 GPUs |

## Getting Started

### Environment

Use a Linux environment with NVIDIA GPUs and CUDA support. The distributed inference entry points use NCCL. From the repository root:

```bash
git clone https://github.com/wyww0626-blip/GeoVP.git
cd GeoVP
conda create -n geovp python=3.10 -y
conda activate geovp
pip install -r requirements.txt
pip install -e .
```

The dependency file pins PyTorch 2.0.1, torchvision 0.15.2, and torchaudio 2.0.2 with a CUDA 11.7 package index. Model-specific dependencies and pretrained visual encoder assets must also be available for the selected model configuration.

### Model checkpoints

The loader expects a checkpoint directory containing the model weights, model configuration, metadata, and tokenizer. It supports `consolidated.*.pth` weights and probes `meta.json`, `config.json`, and the tokenizer files to determine how to instantiate the model. The model type named in the metadata must have a corresponding implementation in `accessory/model/LLM/`.

Checkpoint weights and the full GeoVP-650K dataset are not bundled in this source tree. A public download URL is not specified here.

### Multimodal chat entry point

The existing image-and-text chat interface can be launched with a checkpoint compatible with an included model implementation:

```bash
python -m accessory.demos.multi_turn_mm \
  --pretrained_path /path/to/compatible-checkpoint \
  --gpu_ids 0 \
  --dtype bf16
```

Replace `/path/to/compatible-checkpoint` with the actual checkpoint directory. If metadata or tokenizer files are stored separately, pass `--llama_type`, `--llama_config`, and `--tokenizer_path` explicitly. Multiple GPU IDs enable tensor-parallel inference.

This command launches image-and-text chat. The paper's complete point, box, and free-form prompting workflow is not available in the current release. The current chat script launches Gradio with `share=True`, which creates a shareable link.

### Code organization

| Path | Purpose |
| --- | --- |
| [`accessory/model/`](accessory/model/) | Model definitions, tokenizer, and checkpoint loading |
| [`accessory/data/`](accessory/data/) | Data readers, image transforms, and conversation formatting |
| [`accessory/main_finetune.py`](accessory/main_finetune.py) | Fine-tuning entry point |
| [`accessory/configs/`](accessory/configs/) | Model and data configuration files |
| [`accessory/demos/`](accessory/demos/) | Interactive inference entry points |
| [`accessory/eval_mm/`](accessory/eval_mm/) | Multimodal inference and evaluation utilities |
| [`assets/readme/`](assets/readme/) | Figures extracted from the GeoVP manuscript |

The training configuration above records the paper's experimental settings. Reproducing those experiments additionally requires the matching GeoVP model configuration, training annotations, checkpoints, and benchmark splits.

## Citation

If you use GeoVP in your research, please cite the manuscript:

```bibtex
@misc{yu_geovp,
  title  = {GeoVP: A Unified Visual Prompting Framework for Multi-Source Remote Sensing Image Understanding},
  author = {Yu, Le and Wang, Yuanwen and Qi, Xiaotong},
  note   = {Manuscript}
}
```

## Contact

For research questions, please contact **Yuanwen Wang** at [wyww0626@gmail.com](mailto:wyww0626@gmail.com).
