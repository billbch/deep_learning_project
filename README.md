# Deep Learning Project – Knowledge Distillation under Domain Shift

## Overview

This project investigates whether **Knowledge Distillation (KD)** improves robustness of image classification models under domain shift.

We use a **teacher–student setup**:

- Teacher: ResNet-34
- Student: ResNet-18 (baseline + KD version later)

---

## Project Structure

```text
project/
  data/
  models/
  training/
  evaluation/
  outputs/
```

---

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Training

Train teacher model:

```bash
python -m training.train_teacher
```

Train baseline student:

```bash
python -m training.train_baseline
```

---

## Evaluation

Evaluate trained models:

```bash
python -m evaluation.evaluate
```

---

## Notes

- Current runs use **reduced epochs and batches** (for testing pipeline only)
- Accuracy is therefore low and not meaningful yet
- Full training requires more epochs and preferably GPU

---

## Person A Deliverable

- End-to-end reproducible pipeline
- Data loading (CIFAR-100)
- Teacher and baseline student training
- Checkpoint saving/loading
- Accuracy evaluation

---

## Next Steps

- Implement Knowledge Distillation (KD)
- Add domain shift transformations (noise, blur, brightness)
- Evaluate robustness
