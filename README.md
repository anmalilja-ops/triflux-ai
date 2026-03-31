# triflux-ai
triflux ai XY Dual-Axis MLP for MNIST. Reads images as rows (Y) and columns (X), encodes each via MLP streams, pools embeddings, stacks into “3D” tensor, merges for 10-class output. Adaptive LR and dropout change with train/test accuracy.

XY Dual-Axis MNIST MLP

Author: boki_231
Date: 2026-03-30
Description: Dual-axis MLP that reads MNIST images as rows (Y) and columns (X), encodes via MLPs, stacks embeddings into “3D”, merges for classification. Adaptive learning rate and dropout based on accuracy.

Requirements
Python 3.9+
PyTorch
TensorFlow (for MNIST dataset)
scikit-learn
NumPy

Install dependencies:

pip install torch tensorflow scikit-learn numpy
Running the AI
Save xy_dual_axis_mlp.py (your script) in a folder.
Run the script:
python xy_dual_axis_mlp.py
Script will:
Load MNIST dataset
Standardize data
Build XY dual-axis MLP
Train with adaptive LR + dropout
Save best model as best_xy_model.pt
Monitor output for epoch, train/test accuracy, LR, dropout.
Notes
Batch size: 1024
Epochs: 20000 (can adjust)
Streams: Y (rows), X (columns) → 256-dim each → merged
Merger MLP: 256×2 → [256, 256, 256] → 10 classes
Adaptive LR: LR_START * (1 - test_acc)^LR_EXPONENT
Adaptive dropout: scaled by (1 - train_acc)^DROPOUT_EXPONENT, capped at 0.9
Best model: saved automatically during training
