# PyTorch to TensorRT C++ Conversion

This repository provides a step-by-step guide and code for converting a PyTorch model to a TensorRT engine and running it in C++. The process includes exporting the PyTorch model to the ONNX format, generating a TensorRT engine, and performing inference. In this example, we demonstrate the conversion using a simple convolutional neural network for recognizing MNIST handwritten digits.

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Conversion Steps](#conversion-steps)
- [Running the Inference](#running-the-inference)
- [References](#references)

## Requirements

Before you begin, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch
- ONNX
- CUDA (for GPU support)
- TensorRT

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/albertnew2012/pytorch-onnx-tensorRT.git
   cd pytorch-onnx-tensorRT

2. Create a simple convolutional model in PyTorch and export it to ONNX format
   ```bash
   python3 ./scripts/simpleCNN_gpu.py
<a name="conversion-steps"></a>
## Conversion Steps

