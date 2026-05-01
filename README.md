# Deep Learning Project - Knowledge Distillation under Domain Shift

## Overview

This project studies whether Knowledge Distillation (KD) improves robustness for CIFAR-100 image classification under domain shift.

Models:

- Teacher: ResNet-34
- Baseline student: ResNet-18
- KD student: ResNet-18 trained from teacher soft targets

## Project Structure

```text
project/
  data/
    data_loader.py
    corruptions.py
  models/
  training/
  evaluation/
  outputs/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `test_model.py` fails because `torch` is missing, activate the environment first and reinstall dependencies:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python test_model.py
```

## Quick Checks

Check that the ResNet teacher and student can process CIFAR-sized inputs:

```bash
python test_model.py
```

Check that CIFAR-100 loads correctly:

```bash
python test_data.py
```

## Training

Train the teacher:

```bash
python -m training.train_teacher
```

Train the baseline student:

```bash
python -m training.train_baseline
```

Train the KD student:

```bash
python -m training.train_kd
```

The training scripts now use 100 epochs by default and save checkpoints under `outputs/checkpoints/`.

## Evaluation

Evaluate all available checkpoints on clean CIFAR-100 and corrupted CIFAR-100:

```bash
python -m evaluation.evaluate
```

The evaluation covers:

- clean test data
- gaussian noise
- blur
- brightness shift
- contrast shift
- severities 1, 3, and 5

Missing checkpoints are skipped automatically. Results are saved to:

```text
outputs/results.json
```

## Plots

Generate comparison plots after running evaluation:

```bash
python -m evaluation.plot_results
```

Figures are saved under:

```text
outputs/figures/
```

## Notes

- Full training is much faster with GPU.
- Accuracy from short or interrupted runs is only useful for pipeline testing, not final conclusions.
- Data files and model outputs are ignored by git; source files for loaders and corruptions are tracked.
