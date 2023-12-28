# PyTorch to TensorRT CPP Conversion 
quick tutorial on how to convert pytorch model to tensorRT engine and run it in cpp

This repository provides a step-by-step guide and code to demonstrate how to convert a PyTorch model a TensorRT engine, and then run it in C++. The process includes exporting the PyTorch model to ONNX format, generating a TensorRT engine, and performing inference. Example used here is  a simple convolutional neural network for recognizing MNIST handwritten digits.

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Conversion Steps](#conversion-steps)
- [Running the Inference](#running-the-inference)
- [References](#references)

## Requirements

Before you begin, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- ONNX
- NVIDIA GPU with TensorRT installed
- CMake
- C++ Compiler
- NvInfer (part of TensorRT)

## Getting Started

Clone this repository:

```bash
git clone https://github.com/yourusername/pytorch-to-tensorrt-mnist.git
cd pytorch-to-tensorrt-mnist
