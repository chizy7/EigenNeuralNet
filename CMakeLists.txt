cmake_minimum_required(VERSION 3.10)
project(EigenNeuralNet)

set(CMAKE_CXX_STANDARD 11)

# Include project headers and Eigen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source files
add_executable(EigenNeuralNet main.cpp src/NeuralNetwork.cpp)
