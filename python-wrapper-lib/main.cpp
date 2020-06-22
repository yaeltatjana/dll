#include <iostream>
#include <typeinfo>

#include "mnist_lib.h"
#include "mnist_lib_types.h"
#include "MnistReader.h"
#include "DenseDenseNet.h"


/**
 * Method to test the library for the simple example
 */
void testSimpleExample() {
    MnistLib lib;
    lib.createNet(28 * 28, 32, 0.001);
    lib.displayNet();
    lib.displayDataset();
    // lib.displayDatasetPretty();
    lib.train(5);
    lib.evaluate();

    // compare with simple example in one method
    // std::cout << "=================================================================" << std::endl;
    // doSimpleExample();
}

/**
 * Method to test different activation types and a multi-layer network
 */
void testActivations() {
    // dense : relu
    auto ptr = std::move(createDenseRelu(28 * 28, 32, 0.001));
    displayDenseRelu(ptr);
    trainDenseRelu(ptr, 5);

    // dense: relu -> relu -> softmax
    std::list <size_t> neurals_in = {28 * 28, 28 * 28, 28 * 28};
    std::list <size_t> neurals_out = {32, 32, 32};
    auto rrso = std::move(createDenseRRSo(std::list < size_t > {784, 784, 784}, neurals_out, 0.001));
    displayDenseRRSo(rrso);
    trainDenseRRSo(rrso, 5);
}

int main() {
    // testSimpleExample();
    testActivations();

    return 0;
}