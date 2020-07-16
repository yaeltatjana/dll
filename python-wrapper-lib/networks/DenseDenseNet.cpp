#include <memory>
#include <typeinfo>
#include "DenseDenseNet.h"

DenseDenseNet::DenseDenseNet() : net(std::make_unique<dbn_2dense>()) {}

DenseDenseNet::DenseDenseNet(std::vector <size_t> &nb_input, std::vector <size_t> &nb_output) :
        net(std::make_unique<dbn_2dense>()) {
    net->template layer_get<0>().init_layer(nb_input[0], nb_output[0]);
    net->template layer_get<1>().init_layer(nb_input[1], nb_output[1]);
}

void DenseDenseNet::setLearningRate(double l_rate) {
    net->learning_rate = l_rate;
}

void DenseDenseNet::setLayerSize(size_t layer, size_t input_size, size_t output_size) {
    switch (layer) {
        case 0:
            net->template layer_get<0>().init_layer(input_size, output_size);
            break;
        case 1:
            net->template layer_get<1>().init_layer(input_size, output_size);
            break;
    }
}

void DenseDenseNet::setInitialMomentum(double m) {
    net->initial_momentum = m;
}

void DenseDenseNet::display() {
    net->display();
}

void DenseDenseNet::displayPretty() {
    net->display_pretty();
}

float DenseDenseNet::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void DenseDenseNet::evaluate(MnistReader &ds) {
    net->evaluate(ds.testSet());
}

float DenseDenseNet::fineTune(TextReader &ds, size_t epochs) {
    return net->fine_tune(ds.getImages(), ds.readLabels(), epochs);
}

void DenseDenseNet::evaluate(TextReader &ds) {
    net->evaluate(ds.getImages(), ds.readLabels());
}

void DenseDenseNet::storeWeights(const std::string &file) {
    net->store(file);
}

void DenseDenseNet::loadWeights(const std::string &file) {
    net->load(file);
}