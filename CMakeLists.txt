cmake_minimum_required(VERSION 3.10)
project(EigenNeuralNet)

set(CMAKE_CXX_STANDARD 17)

# Include project headers and Eigen
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/eigen-3.4.0)

# Add source files
add_executable(EigenNeuralNet main.cpp src/NeuralNetwork.cpp)
