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
The conversion process involves the following steps:

1. Export PyTorch Model to ONNX format: This step converts your PyTorch model to the ONNX format, which is compatible with TensorRT.
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=./simple_cnn.onnx --saveEngine=./simple_cnn.engine 
2. Build the C++ Inference Code: Set up the C++ environment for running the TensorRT inference. Build the C++ code using CMake.

3. Generate TensorRT Engine: Use the C++ code to generate a TensorRT engine from the ONNX model.


<a name="running-the-inference"></a>
## Running the Inference

1. Navigate to the src directory
   ```bash
   cd src
2. Create a build directory and build the C++ code
   ```bash
   mkdir build && cd build
   cmake ..
   make
3. Run the TensorRT inference
   ```bash
   ./tensorrt_inference
This program will load the TensorRT engine, perform inference on the MNIST dataset, and display the results.

<a name="references"></a>
## References
- PyTorch
- ONNX
- NVIDIA CUDA Toolkit
- NVIDIA TensorRT



