# Text Simplification - CLEF 2026 SimpleText Task 1

Evaluation framework for biomedical text simplification methods on the Cochrane-auto corpus.

## Project Overview

- **Task**: CLEF 2026 SimpleText Task 1 (Text Simplification)
- **Dataset**: Cochrane-auto corpus (1,544 test sentence pairs)
- **Methods**: Multiple approaches (starting with few-shot prompting baseline)
- **Metrics**: SARI (following LIS at SimpleText 2025 evaluation approach)

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- (Optional) CUDA-capable GPU for faster inference

### Installation

```bash
# Install dependencies
uv sync

# Copy the example .env file
cp .env.example .env

# Edit .env to customize configuration (optional)
# nano .env
```

### Configuration

The project uses a `.env` file for configuration. Default values are provided in `.env.example`.

**Key configuration options:**

```bash
# Model to use (change this to use a different model)
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

# Model parameters
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
LOAD_IN_4BIT=false

# Data and evaluation
DATA_DIR=cochrane/data
BATCH_SIZE=8
RANDOM_SEED=42
```

**To use a different model**, simply edit `.env`:

```bash
# Example: Use a smaller model
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct

# Or use GPT-2
MODEL_NAME=gpt2
```

All scripts will automatically use the model specified in `.env`.

## Implemented Methods

### Method 1: Few-Shot Prompting (Baseline)

Zero-training approach using Mistral 7B Instruct with in-context examples.

**Important**: Uses carefully curated simplification examples from the training data (all showing true length reduction, not expansion).

**Run evaluation**:
```bash
uv run python experiments/sentence_level/run_baseline.py
```

See [`METHODS.md`](METHODS.md) for method details.

### Common Options

**Quick test on a subset** (for development):
```bash
uv run python experiments/sentence_level/run_baseline.py --test_size 100
```

**Different number of shots**:
```bash
uv run python experiments/sentence_level/run_baseline.py --num_shots 5
```

**Use manually curated examples**:
```bash
uv run python experiments/sentence_level/run_baseline.py --use_curated_examples
```

**Enable 4-bit quantization** (saves memory on GPU):
```bash
uv run python experiments/sentence_level/run_baseline.py --load_in_4bit
```

**Custom output directory**:
```bash
uv run python experiments/sentence_level/run_baseline.py --output_dir ./my_results
```

### All Options Combined

```bash
uv run python experiments/sentence_level/run_baseline.py \
  --num_shots 3 \
  --test_size 500 \
  --batch_size 16 \
  --load_in_4bit \
  --use_curated_examples \
  --output_dir experiments/sentence_level/results/run1
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_shots` | int | 3 | Number of few-shot examples |
| `--test_size` | int | None (all) | Limit test set size |
| `--batch_size` | int | 8 | Batch size for generation |
| `--output_dir` | str | experiments/sentence_level/results | Output directory |
| `--data_dir` | str | cochrane/data | Data directory |
| `--load_in_4bit` | flag | False | Use 4-bit quantization |
| `--use_curated_examples` | flag | False | Use curated examples |
| `--seed` | int | 42 | Random seed |

## Evaluation Metrics

This project follows the evaluation approach from **LIS at SimpleText 2025** (Paper 358), using **SARI** as the primary metric.

### SARI Score

SARI (System output Against References and Input) measures simplification quality by analyzing:
- **Additions**: n-grams added in the simplification
- **Deletions**: n-grams removed from the source
- **Keeps**: n-grams retained from the source

**Score Range**: 0-100 (higher is better)

**Interpretation**:
- **<25**: Poor simplification
- **25-35**: Acceptable baseline
- **35-42**: Good system
- **>42**: State-of-the-art

**Reference Benchmark**:
- LIS at SimpleText 2025 best result: **43.51** (5th place at CLEF 2025)
- Model: Mistral 7B with zero-shot prompting + medical definitions (MedSimplify glossary)

See [`METRICS.md`](METRICS.md) for additional metric details.

## Output Files

After running evaluation, find results in the output directory:

- **`predictions.txt`**: Generated simplifications (one per line)
- **`input_output_pairs.txt`**: Side-by-side comparison of input/output/references
- **`metrics.json`**: SARI score with metadata

### Example metrics.json

```json
{
  "metrics": {
    "sari": 43.5051
  },
  "metadata": {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "num_shots": 3,
    "test_size": 1544,
    "timestamp": "2026-05-04T17:30:00",
    "evaluation": "SARI-only (following Paper 358: LIS at SimpleText 2025)"
  }
}
```

## Paper 358 Prompting Strategies

This project includes implementations of the prompting strategies from **LIS at SimpleText 2025** (Paper 358), which achieved **SARI 43.51** at CLEF 2025.

The paper's approach uses:
- **Keyword extraction** for domain-specific term identification
- **Definition-augmented prompts** with medical glossary (MedSimplify)
- **Zero-shot prompting** (best: SARI 43.51)
- **Iterative refinement** for post-processing

See [`PAPER358_PROMPTS.md`](PAPER358_PROMPTS.md) for detailed documentation and usage examples.

### Quick Example

```python
from src.prompts.paper358_prompts import get_zero_shot_prompt

definitions = [
    {"term": "intubation", "definition": "insertion of a breathing tube"},
    {"term": "DR", "definition": "delivery room"}
]

prompt = get_zero_shot_prompt(complex_text, definitions)
# Use with your LLM for simplification
```

Test all prompt strategies:
```bash
uv run python test_paper358_prompts.py
```

## Performance Targets

Target scores for different methods documented in [`METHODS.md`](METHODS.md).

## Data Structure

The Cochrane test data (`cochrane/data/cochraneauto_sents_test.csv`) contains:

- **`complex`**: Input sentence to simplify
- **`simple`**: Reference simplified sentence(s) (list format)
- **`label`**: Simplification operation (rephrase/split/delete/merge)
- **`pair_id`**: Document identifier
- **Additional metadata**: paragraph ID, sentence position, etc.

## Project Structure

```
simple_text/
├── src/
│   ├── models/
│   │   └── llama_simplifier.py    # Model wrapper for text simplification
│   ├── evaluation/
│   │   └── metrics.py             # SARI, BLEU, BERTScore, LENS
│   ├── prompts/
│   │   ├── few_shot_examples.py   # Curated examples
│   │   └── templates.py           # Prompt templates
│   └── utils/
│       └── data_loader.py         # Load Cochrane data
├── experiments/
│   └── sentence_level/
│       ├── run_baseline.py        # Main evaluation script
│       └── results/               # Output directory
├── cochrane/                      # Cochrane-auto corpus
│   └── data/
│       ├── cochraneauto_sents_test.csv
│       ├── cochraneauto_sents_train.csv
│       └── ...
├── pyproject.toml                 # uv dependencies
├── .python-version                # Python 3.12
├── README.md                      # This file
└── METHODS.md                     # Detailed methodology
```

## Adding New Methods

To add a new simplification method:

1. **Create model/approach** in `src/models/`
2. **Add evaluation script** in `experiments/sentence_level/`
3. **Document method** in `METHODS.md`
4. **Run evaluation** and compare results

## Future Work

- Fine-tuning approaches
- Document-level simplification
- Additional evaluation metrics

## Troubleshooting

### Out of Memory

**Problem**: Script uses too much RAM / CUDA out of memory error

**Memory Usage by Model:**
| Model | RAM (FP16) | RAM (4-bit) | Quality |
|-------|------------|-------------|---------|
| Llama-3.2-1B | ~2GB | ~1GB | Good |
| Llama-3.2-3B | ~6GB | ~2GB | Better |
| Mistral-7B | ~14GB | ~4GB | Best |

**Solutions:**

1. **Enable 4-bit quantization** (reduces memory by 70%):
   ```bash
   # In .env file
   LOAD_IN_4BIT=true
   ```
   Or use command line flag:
   ```bash
   uv run python experiments/sentence_level/simplify_sentence.py "text" --load_in_4bit
   ```

2. **Use a smaller model** - Edit `.env`:
   ```bash
   # 1B model uses only ~1-2GB RAM
   MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
   ```

3. **Run on CPU** (slower but no GPU memory needed):
   ```bash
   CUDA_VISIBLE_DEVICES="" uv run python experiments/sentence_level/simplify_sentence.py "text"
   ```

4. **Reduce batch size** (for run_baseline.py):
   ```bash
   uv run python experiments/sentence_level/run_baseline.py --batch_size 1
   ```

### Slow Generation

**Problem**: Generation is very slow

**Solutions**:
- Use GPU if available
- Reduce test size for development: `--test_size 100`
- Increase batch size: `--batch_size 16`

### Import Errors

**Problem**: Module not found errors

**Solution**:
```bash
uv sync  # Reinstall all dependencies
```

### LENS Not Available

**Problem**: LENS metric cannot be computed

**Note**: LENS requires separate installation from GitHub. The baseline will run without it, computing SARI, BLEU, and BERTScore only.

## Citation

If you use this baseline, please cite:

```bibtex
@inproceedings{cochrane-auto-2024,
  title={Cochrane-auto: An Aligned Dataset for the Simplification of Biomedical Abstracts},
  author={Devaraj et al.},
  booktitle={TSAR Workshop},
  year={2024}
}
```

## Documentation

- **[METHODS.md](METHODS.md)**: Simplification methods and approaches
- **[METRICS.md](METRICS.md)**: Evaluation metrics and formulas
- **[Cochrane README](cochrane/README.md)**: Dataset documentation

## License

This project is for research and educational purposes as part of CLEF 2026 SimpleText shared task.
