#include "AlexNet.h"

AlexNet::AlexNet() : net(std::make_unique<dbn_alexnet>()) {}

void AlexNet::display() {
    net->display();
}

void AlexNet::displayPretty() {
    net->display_pretty();
}

void AlexNet::setConvLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t nbFilt, size_t filt1, size_t filt2) {
    switch(layer) {
        case 0:net->template layer_get<0>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 2:net->template layer_get<2>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 4:net->template layer_get<4>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 5:net->template layer_get<5>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 6:net->template layer_get<6>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
    }
}

void AlexNet::setMPLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t pool1, size_t pool2) {
    switch(layer) {
        case 1: net->template layer_get<1>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 3: net->template layer_get<3>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 7: net->template layer_get<7>().init_layer(channel, dim1, dim2, pool1, pool2); break;
    }
}

void AlexNet::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    switch(layer) {
        case 8: net->template layer_get<8>().init_layer(inputSize, outputSize); break;
        case 9: net->template layer_get<9>().init_layer(inputSize, outputSize); break;
    }
}

void AlexNet::setLearningRate(double rate) {
    net->learning_rate = rate;
}

void AlexNet::setAdamBeta1(double beta) {
    net->adam_beta1 = beta;
}

void AlexNet::setAdamBeta2(double beta) {
    net->adam_beta2 = beta;
}

float AlexNet::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void AlexNet::evaluate(MnistReader &ds) {
    net->evaluate(ds.trainSet());
}

void AlexNet::storeWeights(const std::string &file) {
    net->store(file);
}

void AlexNet::loadWeights(const std::string &file) {
    net->load(file);
}
