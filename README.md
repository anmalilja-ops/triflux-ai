Triflux AI — 3D Neural Network Digit Recognition
Triflux AI is a custom-built 3D neural network for handwritten digit recognition, trained on the MNIST dataset. Unlike traditional flat (2D) neural networks, Triflux processes data through a three-dimensional architecture, achieving ~99% accuracy. XL is hitting up to 99% n digit classification.

Models
ModelFileSizeBest ForTriflux XLai_V9_trifulx-xl-gen5_MNIST.pyExtra LargeMaximum accuracyTriflux Mai_V9_trifulx-m-gen5_MNIST.pyMediumFaster inference, lower resource use
Both models are Generation 5 (gen5) — the latest version of the Triflux architecture.

What makes Triflux different?
Most digit recognition models use standard 2D neural networks (flat layers). Triflux uses a 3D neural network architecture, which means:

Data flows through three-dimensional layers
More complex feature relationships can be captured
A unique approach to a classic AI benchmark problem


Dataset
Trained on MNIST — 70,000 handwritten digit images (0–9), the standard benchmark for digit recognition.

Training set: 60,000 images
Test set: 10,000 images
Accuracy: ~99%


Getting Started
Requirements
bashpip install -r requirements.txt
Run the XL model XL is hitting up to 99%
bashpython ai_V9_trifulx-xl-gen5_MNIST.py
Run the Medium model
bashpython ai_V9_trifulx-m-gen5_MNIST.py

Project Structure
├── ai_V9_trifulx-xl-gen5_MNIST.py   # Triflux XL model
├── ai_V9_trifulx-m-gen5_MNIST.py    # Triflux Medium model
└── README.md

License
This project is open source. Feel free to use, modify, and build on it.
