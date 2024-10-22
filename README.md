# EigenNeuralNet

Simple C++ neural network using Eigen with customizable architectures and activation functions.

## Features

- **Customizable Network Architecture**: Easily adjust the number of layers and neurons.
- **Activation Functions**: Supports Sigmoid and Tanh activation functions.
- **Training and Prediction**: Includes methods for training the network and making predictions.
- **Example Usage**: Comes with an example solving the XOR problem.
- **Modular Code Structure**: Organized for easy understanding and extension.

## Requirements

- **C++ Compiler**: Compatible with C++11 or higher.
- **Eigen Library**: [Eigen](https://eigen.tuxfamily.org/)
- **CMake**: Build system generator (version 3.10 or higher).

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/chizy7/EigenNeuralNet.git
cd EigenNeuralNet
```
### Building the Project

#### Prerequisites

Ensure that you have the following installed:

- **CMake**: Install via Homebrew on macOS (`brew install cmake`) or Chocolatey on Windows or from the [official website](https://cmake.org/download/).
- **Eigen Library**: Download and extract the Eigen library to the `EigenNeuralNet` directory.

### Steps to Build

1. Create a build directory and navigate to it:

```bash
mkdir build
cd build
```
2. Run CMake Configuration to generate the build files:

```bash
cmake ..
```
3. Build the project:

```bash
make
```

### Running the Program

After building, run the executable:

```bash 
./EigenNeuralNet
```

### Expected Output

The program will train the neural network to solve the XOR problem and output the predictions.

```bash
Input: 0 0 Output: 0.01
Input: 0 1 Output: 0.99
Input: 1 0 Output: 0.99
Input: 1 1 Output: 0.01
```
This output demonstrates that the neural network has successfully learned the XOR function.

## Customization

### Changing the Network Architecture

Modify the architecture vector in main.cpp:

```cpp
// Example: 2 inputs, 3 hidden neurons, 1 output
std::vector<int> architecture = {2, 3, 1};
```

### Switching Activation Functions

Change the activation function parameter when instantiating the neural network:

```cpp
NeuralNetwork nn(architecture, 0.1, NeuralNetwork::TANH);
```

### Adjusting Learning Rate and Epochs

Modify the learning rate and number of training epochs in main.cpp:

```cpp
NeuralNetwork nn(architecture, 0.05, NeuralNetwork::SIGMOID);

// Training loop
for (int epoch = 0; epoch < 5000; ++epoch) {
    // Training code...
}
```

## Code Overview

### NeuralNetwork Class

Located in `include/NeuralNetwork.hpp` and `src/NeuralNetwork.cpp` the NeuralNetwork class provides the core functionality:

- **Constructor**: Initializes the network with the specified architecture and activation function.
- **train**: Trains the network using input and target output vectors.
- **predict**: Generates predictions for given input data.

### Main Program

The `main.cpp` file demonstrates how to use the NeuralNetwork class to solve the XOR problem.

## Medium Article

For a detailed explanation and step-by-step guide on implementing this neural network, please refer to my Medium article: [Implementing a Simple Neural Network in C++ with Eigen](https://medium.com/@chizy7/implementing-a-simple-neural-network-in-c-with-eigen-f555a664f7b8)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.