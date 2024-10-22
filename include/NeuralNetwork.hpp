#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork {
public:
    enum ActivationFunction { TANH, SIGMOID };

    NeuralNetwork(const std::vector<int>& architecture,
                    double learning_rate = 0.01,
                    ActivationFunction activation = TANH);

    void train(const Eigen::VectorXd& input, const Eigen::VectorXd& target);
    Eigen::VectorXd predict(const Eigen::VectorXd& input);

private:
    double learning_rate;
    ActivationFunction ActivationFunction;
    std::vector<int> architecture;
    std::vector<Eigen::VectorXd> neurons;
    std::vector<Eigen::VectorXd> deltas;
    std::vector<Eigen::MatrixXd> weights;

    void initializeWeights();
    void forward(const Eigen::VectorXd& input);
    void backward(const Eigen::VectorXd& target);
    double activation(double x);
    double activationDerivative(double x);
}



#endif // NEURAL_NETWORK_HPP

