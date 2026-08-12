# Triflux AI — Dual-Axis Neural Network Digit Recognition

Triflux AI is a custom neural-network architecture for handwritten digit recognition on the MNIST dataset.

Instead of treating an image only as a flat vector, Triflux processes the image through **two independent spatial directions**:

* **Y-stream:** processes the 28 rows of the image
* **X-stream:** processes the 28 columns of the image
* The two representations are combined and passed through a merger network for classification

This creates a **3D-style combined representation** from the two spatial processing streams.

## Performance

Triflux can reach approximately **99% MNIST digit-classification accuracy**, depending on the model version and training configuration.

## Architecture

For each 28×28 MNIST image:

```text
                 MNIST Image
                     │
              ┌──────┴──────┐
              │             │
          Y-stream       X-stream
          28 rows        28 columns
              │             │
         Row encoder     Column encoder
              │             │
          32-dim          32-dim
         embedding       embedding
              │             │
              └──────┬──────┘
                     │
              Combined embedding
                     │
                Merger network
                     │
                 10 classes
```

The current architecture uses:

* 28×28 MNIST input
* Separate X and Y processing streams
* Three 64-unit hidden layers in each stream
* 32-dimensional output embedding per stream
* Three 64-unit merger layers
* Batch normalization
* ReLU activation
* Adaptive dropout
* Adaptive learning rate

The model code reshapes the input into a 28×28 representation, independently encodes rows and columns, mean-pools each stream, and concatenates the resulting embeddings before classification.

## Adaptive Training

Triflux also uses adaptive training mechanisms.

### Adaptive learning rate

The learning rate changes according to test accuracy:

```text
LR = LR_START × (1 - test_accuracy)^LR_EXPONENT
```

### Adaptive dropout

Dropout changes according to training accuracy:

```text
Dropout = DROPOUT_SCALE × (1 - train_accuracy)^DROPOUT_EXPONENT
```

The dropout value is smoothed over several epochs rather than changing instantly.

## Dataset

Triflux is trained and evaluated on **MNIST**:

* 60,000 training images
* 10,000 test images
* 28×28 grayscale images
* 10 digit classes (0–9)

The current implementation loads the complete MNIST dataset and standardizes the input features before training.

## Generation

The current implementation is **Triflux Gen 9**.

Gen 9 is a dual-axis, 3D-stacked architecture with adaptive learning-rate and dropout mechanisms.

## Why Triflux?

The goal of Triflux is to explore whether neural networks can become more efficient by explicitly processing different spatial directions and combining their representations instead of relying entirely on a conventional flat architecture.

It is a small experimental architecture, but the same general idea — **extracting structured representations before combining them** — could potentially be explored in much larger AI systems.

## Running Triflux

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the desired model:

```bash
python ai_V9_trifulx-xl-gen5_MNIST.py
```

or:

```bash
python ai_V9_trifulx-m-gen5_MNIST.py
```

## Project Structure

```text
├── ai_V9_trifulx-xl-gen5_MNIST.py
├── ai_V9_trifulx-m-gen5_MNIST.py
└── README.md
```

## License

This project is open source. Feel free to use, modify, and build on it.
