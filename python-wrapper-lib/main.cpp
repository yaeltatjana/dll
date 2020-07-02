#include <iostream>

#include "mnist_lib.h"
#include "mnist_lib_types.h"
#include "datasets/MnistReader.h"
#include "datasets/Mnist3DReader.h"
#include "datasets/TextReader.h"
#include "networks/DenseDenseNet.h"
#include "networks/DenseDenseDenseNet.h"
#include "networks/LeNet.h"
#include "networks/AlexNet.h"
#include "networks/VGGNet.h"


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
    net2.storeWeights("test_store.txt");
}

void testDDDNet() {
    MnistReader ds;
    // layers not initialized with constructor
    DenseDenseDenseNet net1;
    net1.setLayerSize(0, 28 * 28, 28 * 28);
    net1.setLayerSize(1, 28 * 28, 30);
    net1.setLayerSize(2, 30, 10);

    net1.display();
    // n.fineTune(ds,5);

    // layer initialized with constructor
    std::vector <size_t> in{28 * 28, 16, 16};
    std::vector <size_t> out{16, 16, 10};
    DenseDenseDenseNet net2(in, out);
    // setLearningRate(0.2);
    // net.setInitialMomentum(0.85);
    net2.display();
    net2.fineTune(ds, 10);
    net2.loadWeights("test_store.txt");
    net2.evaluate(ds);
    net2.storeWeights("test_store.txt");
}

void testLeNet() {
    LeNet n;
    n.display();
    n.displayPretty();

    //Mnist3DReader r;
    //n.fineTune(r,20);
}

void testMnist3DReader() {
    Mnist3DReader r;
    std::cout << r.readTrainingImages().size() << std::endl;
    std::cout << r.readTestImages().size() << std::endl;
    std::cout << r.readTrainingLabels().size() << std::endl;
    std::cout << r.readTestLabels().size() << std::endl;
}

void testAlexNet() {
    AlexNet n;
    n.setConvLayer(0, 1, 28, 28, 12, 5, 5);
    n.setMPLayer(1, 12, 24, 24, 2, 2);
    n.setConvLayer(2, 12, 12, 12, 12, 3, 3);
    n.setMPLayer(3, 12, 10, 10, 2, 2);
    n.setConvLayer(4, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(5, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(6, 12, 5, 5, 12, 2, 2);
    n.setMPLayer(7, 12, 4, 4, 2, 2);
    n.setDenseLayer(8, 12 * 2 * 2, 32);
    n.setDenseLayer(9, 32, 10);
    n.display();

    MnistReader r;
    n.fineTune(r, 5);
    n.evaluate(r);
}

void testVGGNet() {
    // TODO: change values, random ones actually
    VGGNet n;
    n.setConvLayer(0, 1, 28, 28, 12, 5, 5);
    n.setConvLayer(1, 1, 28, 28, 12, 5, 5);
    n.setMPLayer(2, 12, 24, 24, 2, 2);

    n.setConvLayer(3, 12, 12, 12, 12, 3, 3);
    n.setConvLayer(4, 12, 12, 12, 12, 3, 3);
    n.setMPLayer(5, 12, 10, 10, 2, 2);

    n.setConvLayer(6, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(7, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(8, 12, 5, 5, 12, 2, 2);
    n.setMPLayer(9, 12, 4, 4, 2, 2);

    n.setConvLayer(10, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(11, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(12, 12, 5, 5, 12, 2, 2);
    n.setMPLayer(13, 12, 4, 4, 2, 2);

    n.setConvLayer(14, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(15, 12, 5, 5, 12, 1, 1);
    n.setConvLayer(16, 12, 5, 5, 12, 2, 2);
    n.setMPLayer(17, 12, 4, 4, 2, 2);

    n.setDenseLayer(18, 12 * 2 * 2, 32);
    n.setDenseLayer(19, 12 * 2 * 2, 32);
    n.setDenseLayer(20, 32, 10);
    n.display();

    MnistReader r;
    n.fineTune(r, 5);
    n.evaluate(r);

}

void testTextReader() {
    TextReader r("test/text_db/images", "test/text_db/labels");
    std::cout << (int) r.readLabels()[0] << std::endl;
}

int main() {
    // testSimpleExample();
    // testActivations();
    // testMnistReader();
    // testDenseDenseNet();
    // testDDDNet();
    // testLeNet();
    // testMnist3DReader();
    // testLeNet();
    // testAlexNet();
    // testVGGNet();
    testTextReader();

    return 0;
}