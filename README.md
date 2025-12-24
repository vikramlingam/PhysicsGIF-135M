# PhysicsGIF

🎬 **Text-to-Physics GIF Generator**: Generate physically-accurate animated GIFs from natural language descriptions. (Experiment only)

## Overview

PhysicsGIF converts text descriptions like "a red ball bouncing" into animated GIFs with realistic physics simulation.

```
"a red ball bouncing to the right" → 🎬 output.gif
```

### How It Works

```
Text Prompt → LLM Parser → JSON Scene Spec → Physics Engine → Renderer → GIF
```

1. **PhysicsGIF-135M** (fine-tuned LLM) parses text to structured JSON
2. **Physics Engine** simulates Newtonian motion
3. **Renderer** draws frames and saves as GIF

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/vikramlingam/PhysicsGIF-135M
cd PhysicsGIF-135M
```

### 2. Install Dependencies

```bash
pip install torch transformers peft pillow numpy tqdm matplotlib
```

### 3. Download Model from Hugging Face

```bash
# Option A: Using huggingface-cli
huggingface-cli download vikramlingam/PhysicsGIF-135M --local-dir models/PhysicsGIF-135M

# Option B: Using git lfs
git lfs install
git clone https://huggingface.co/vikramlingam/PhysicsGIF-135M models/PhysicsGIF-135M
```

### 4. Generate GIFs

**Interactive Mode:**
```bash
python generate.py
```

```
🎬 PhysicsGIF Text-to-GIF Generator
==================================================
Enter prompt: a red ball bouncing to the right
Generating...
✓ Generated: output_1.gif

Enter prompt: two triangles colliding and exploding
Generating...
✓ Generated: output_2.gif

Enter prompt: quit
Goodbye! 👋
```

**Single Command:**
```bash
python generate.py "a blue square falling down" -o my_animation.gif
```

## Supported Prompts

### Objects
- `ball`, `circle`, `sphere`
- `square`, `box`, `cube`
- `triangle`, `pyramid`

### Colors
`red`, `blue`, `green`, `yellow`, `orange`, `purple`, `pink`, `cyan`, `white`

### Motion Types
- `bouncing` — Gravity + elastic bounce
- `falling`, `dropping` — Falls from top
- `floating`, `hovering` — No gravity
- `colliding`, `crashing` — Objects collide
- `exploding`, `blasting` — Particle explosion effects

### Multi-Object
- `two balls colliding`
- `three triangles bouncing`

### Examples
```bash
python generate.py "a red ball bouncing"
python generate.py "blue square falling down"
python generate.py "two green triangles colliding and exploding"
python generate.py "small purple ball floating left"
```

## Project Structure

```
PhysicsGIF/
├── generate.py              # Main CLI (interactive mode)
├── requirements.txt
├── README.md
├── src/
│   ├── dsl.py               # Scene specification DSL
│   ├── parser.py            # LLM text parser
│   ├── physics.py           # Physics simulation engine
│   ├── renderer.py          # GIF renderer
│   └── pipeline.py          # Full pipeline
└── models/
    └── PhysicsGIF-135M/     # Download from HuggingFace
```

## Model Details

| Metric | Value |
|--------|-------|
| Base Model | SmolLM2-135M-Instruct |
| Fine-tuning | LoRA (r=16, alpha=32) |
| Training Examples | 500 |
| Epochs | 20 |
| Final Loss | 0.092 |
| Training Time | 42 minutes (CPU) |

**Model Card:** [huggingface.co/vikramlingam/PhysicsGIF-135M](https://huggingface.co/vikramlingam/PhysicsGIF-135M)

## CLI Options

```bash
python generate.py [TEXT] [OPTIONS]

Arguments:
  TEXT                Text description (optional, runs interactive mode if not provided)

Options:
  -o, --output PATH   Output GIF path (default: output.gif)
  -m, --model PATH    Model path (default: models/PhysicsGIF-135M)
  --no-model          Use rule-based parser instead of LLM
  -q, --quiet         Suppress progress messages
```

## Training Your Own Model

```bash
# Generate training data
python generate_dataset.py

# Train with visualizations
python train_parser.py --epochs 20 --output models/my_model
```

Training generates 12 visualization charts in `models/my_model/visualizations/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 PhysicsGIF Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "a red ball bouncing"                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                    │
│  │  PhysicsGIF-135M    │  (Fine-tuned LLM)                 │
│  │  Text → JSON DSL    │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │  physics.py         │  (Newtonian simulation)           │
│  │  Euler integration  │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │  renderer.py        │  (PIL-based rendering)            │
│  │  Frame → GIF        │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│         output.gif                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Pillow
- NumPy

## License

Apache 2.0

## Citation

```bibtex
@misc{physicsgif2024,
  title={PhysicsGIF: Text-to-Physics Animation via Fine-tuned Language Models},
  author= Vikram Lingam,
  year={2025},
  url={https://github.com/vikramlingam/PhysicsGIF-135M}
}
```
