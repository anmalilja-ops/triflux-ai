Author & Date

boki_231 — 2026-03-30

Requirements
Python 3.9+
PyTorch
TensorFlow (for MNIST dataset)
scikit-learn
NumPy

Install dependencies:

pip install torch tensorflow scikit-learn numpy
Running the AI

Example scripts:

ai_V9_trifulx-xl-gen5_MNIST.py
ai_V9_trifulx-m-gen5_MNIST.py

Run:

py -3.11 ai_V9_trifulx_gen1_MNIST.py

Script will:

Load MNIST dataset
Standardize data
Build XY dual-axis MLP
Train with adaptive LR + dropout
Save best model as best_xy_model.pt
Notes
Batch size: 1024
Epochs: 20000 (adjustable)
Streams: Y (rows), X (columns) → 256-dim each → merged
Merger MLP: 256×2 → [256, 256, 256] → 10 classes
Adaptive LR: LR_START * (1 - test_acc)^LR_EXPONENT
Adaptive dropout: scaled by (1 - train_acc)^DROPOUT_EXPONENT, capped at 0.9
Best model: automatically saved during training
