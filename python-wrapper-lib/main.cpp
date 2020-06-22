#include <iostream>

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


/**
 * Method to test the MNIST reader library
 */
void testMnistReader() {
    MnistReader ds;
    ds.display();
    ds.displayPretty();
    ds.trainSet();
    //ds.validationSet();
    ds.testSet().display();
}

/**
 * Method to test the network with dense layers (relu -> softmax)
 */
void testDenseDenseNet() {
    MnistReader ds;
    // layers not initialized with constructor
    DenseDenseNet net1;
    net1.setLayerSize(0, 28 * 28, 28 * 28);
    net1.setLayerSize(1, 28 * 28, 30);

    // n.display();
    // n.fineTune(ds,5);
    //n.all();

    // layer initialized with constructor
    std::vector <size_t> in{28 * 28, 16};
    std::vector <size_t> out{16, 10};
    DenseDenseNet net2(in, out);
    // setLearningRate(0.2);
    // net.setInitialMomentum(0.85);
    net2.display();
    net2.fineTune(ds, 5);
    net2.evaluate(ds);
}

int main() {
    // testSimpleExample();
    // testActivations();
    // testMnistReader();
    testDenseDenseNet();

    return 0;
}