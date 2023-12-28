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
Create a simple convolutional model in PyTorch and export it to ONNX format:
  ````bash
  python3 ./scripts/simpleCNN_gpu.py

## conversion-steps
Conversion Steps
The conversion process involves the following steps:

Export PyTorch Model to ONNX format: This step converts your PyTorch model to the ONNX format, which is compatible with TensorRT.

Build the C++ Inference Code: Set up the C++ environment for running the TensorRT inference. Build the C++ code using CMake.

Generate TensorRT Engine: Use the C++ code to generate a TensorRT engine from the ONNX model.

Running the Inference
Navigate to the src directory:

bash
Copy code
cd src
Create a build directory and build the C++ code:

bash
Copy code
mkdir build && cd build
cmake ..
make
Run the TensorRT inference:

bash
Copy code
./tensorrt_inference
This program will load the TensorRT engine, perform inference on the MNIST dataset, and display the results.

References
PyTorch
ONNX
NVIDIA CUDA Toolkit
NVIDIA TensorRT
