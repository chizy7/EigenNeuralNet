#include "NeuralNetwork.hpp"
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& architecture,
                             double learningRate,
                             ActivationFunction activation)
    : learningRate(learningRate), activationFunction(activation), architecture(architecture) {
    // Initialize neurons and deltas
    for (size_t i = 0; i < architecture.size(); ++i) {
        int layerSize = architecture[i] + (i < architecture.size() - 1 ? 1 : 0); // Add bias neuron
        neurons.emplace_back(Eigen::VectorXd::Zero(layerSize));
        deltas.emplace_back(Eigen::VectorXd::Zero(layerSize));
        if (i < architecture.size() - 1) {
            neurons[i](layerSize - 1) = 1.0; // Set bias neuron activation to 1
        }
    }
    // Initialize weights
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        int rows = neurons[i].size();
        int cols = neurons[i + 1].size() - 1; // Exclude bias neuron in next layer
        Eigen::MatrixXd weightMatrix = Eigen::MatrixXd::Random(rows, cols);
        weights.push_back(weightMatrix);
    }
}

double NeuralNetwork::activate(double x) {
    if (activationFunction == SIGMOID) {
        return 1.0 / (1.0 + std::exp(-x));
    } else { // TANH
        return std::tanh(x);
    }
}

double NeuralNetwork::activateDerivative(double x) {
    if (activationFunction == SIGMOID) {
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    } else { // TANH
        double tanhVal = std::tanh(x);
        return 1 - tanhVal * tanhVal;
    }
}

void NeuralNetwork::forward(const Eigen::VectorXd& input) {
    neurons[0].head(architecture[0]) = input;

    for (size_t i = 1; i < architecture.size(); ++i) {
        Eigen::VectorXd netInput = weights[i - 1].transpose() * neurons[i - 1];
        for (int j = 0; j < architecture[i]; ++j) {
            neurons[i](j) = activate(netInput(j));
        }
        if (i < architecture.size() - 1) {
            neurons[i](architecture[i]) = 1.0; // Set bias neuron
        }
    }
}

void NeuralNetwork::backward(const Eigen::VectorXd& target) {
    // Output layer deltas
    Eigen::VectorXd netInput = weights.back().transpose() * neurons[neurons.size() - 2];
    for (int i = 0; i < architecture.back(); ++i) {
        double output = neurons.back()(i);
        double error = output - target(i);
        deltas.back()(i) = error * activateDerivative(netInput(i));
    }

    // Hidden layer deltas
    for (int i = architecture.size() - 2; i > 0; --i) {
        Eigen::VectorXd netInput = weights[i - 1].transpose() * neurons[i - 1];
        Eigen::VectorXd delta = weights[i].block(0, 0, weights[i].rows(), architecture[i + 1]) * deltas[i + 1].head(architecture[i + 1]);
        for (int j = 0; j < architecture[i]; ++j) {
            deltas[i](j) = delta(j) * activateDerivative(netInput(j));
        }
    }

    // Update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        Eigen::MatrixXd deltaWeight = neurons[i] * deltas[i + 1].head(architecture[i + 1]).transpose();
        weights[i] -= learningRate * deltaWeight;
    }
}

void NeuralNetwork::train(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
    forward(input);
    backward(target);
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
    forward(input);
    return neurons.back().head(architecture.back());
}
