#ifndef DLL_MNIST_LIB_H
#define DLL_MNIST_LIB_H

#include <vector>
#include <memory>
#include <list>
#include "dll/neural/dense_layer.hpp"
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"
#include "mnist_lib_types.h"

/**
 * Class used to create a neural network and to train/evaluate it on MNIST dataset
 */
class MnistLib {
private:
    struct Dataset ds;
    std::unique_ptr <dbn_t> net_dbn;
    ds_t dataset;

public:
    MnistLib();

    struct Dataset getDataset();

    void createNet(size_t nb_input, size_t nb_output, double learning_rate);

    void displayDataset();

    void displayDatasetPretty();

    void displayNet();

    float train(size_t epochs);

    void evaluate();

};

/**
 * Function to launch the simple example
 */
void doSimpleExample();

ds_t getMNISTDataset();

// dense layer with relu activation
std::unique_ptr<dbn_relu>& createDenseRelu(size_t nb_input, size_t nb_output, double learning_rate);
void displayDenseRelu(std::unique_ptr<dbn_relu>& net);
float trainDenseRelu(std::unique_ptr<dbn_relu>& net, size_t epochs);

// 3x dense layers : relu -> relu -> softmax
std::unique_ptr<dbn_RRSo>& createDenseRRSo(std::list<size_t> nb_input, std::list<size_t> nb_output, size_t learning_rate);
void displayDenseRRSo(std::unique_ptr<dbn_RRSo>& net);
float trainDenseRRSo(std::unique_ptr<dbn_RRSo>& net, size_t epochs);
void allRRSo();

#endif //DLL_MNIST_LIB_H
