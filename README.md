# triflux-ai
triflux ai XY Dual-Axis MLP for MNIST. Reads images as rows (Y) and columns (X), encodes each via MLP streams, pools embeddings, stacks into “3D” tensor, merges for 10-class output. Adaptive LR and dropout change with train/test accuracy.
