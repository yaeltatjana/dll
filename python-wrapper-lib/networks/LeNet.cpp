#include <memory>

#include "LeNet.h"

LeNet::LeNet() : net(std::make_unique<dbn_lenet>()) { }

void LeNet::display() {
    net->display();
}

void LeNet::displayPretty() {
    net->display_pretty();
}

void LeNet::setConvLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t nbFilt, size_t filt1, size_t filt2) {

    switch(layer) {
        case 0:net->template layer_get<0>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 2:net->template layer_get<2>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
    }
}

void LeNet::setMPLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t pool1, size_t pool2) {
    switch(layer) {
        case 1: net->template layer_get<1>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 3: net->template layer_get<3>().init_layer(channel, dim1, dim2, pool1, pool2); break;
    }
}

void LeNet::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    switch(layer) {
        case 4: net->template layer_get<4>().init_layer(inputSize, outputSize); break;
        case 5: net->template layer_get<5>().init_layer(inputSize, outputSize); break;
    }
}

void LeNet::setLearningRate(double rate) {
    net->learning_rate = rate;
}

void LeNet::setAdamBeta1(double beta) {
    net->adam_beta1 = beta;
}

void LeNet::setAdamBeta2(double beta) {
    net->adam_beta2 = beta;
}

float LeNet::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void LeNet::evaluate(MnistReader &ds) {
    net->evaluate(ds.trainSet());
}

void LeNet::storeWeights(const std::string &file) {
    net->store(file);
}

void LeNet::loadWeights(const std::string &file) {
    net->load(file);
}
