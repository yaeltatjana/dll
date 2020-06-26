#include <memory>
#include "DenseDenseNet.h"

DenseDenseNet::DenseDenseNet() : net(std::make_unique<dbn_dense_RSo>()) {}

DenseDenseNet::DenseDenseNet(std::vector <size_t> &nb_input, std::vector <size_t> &nb_output) :
        net(std::make_unique<dbn_dense_RSo>()) {
    net->template layer_get<0>().init_layer(nb_input[0], nb_output[0]);
    net->template layer_get<1>().init_layer(nb_input[1], nb_output[1]);
}

void DenseDenseNet::setLearningRate(double l_rate) {
    net->learning_rate = l_rate;
}

void DenseDenseNet::setLayerSize(size_t layer, size_t input_size, size_t output_size) {
    if (layer == 0) {
        net->template layer_get<0>().init_layer(input_size, output_size);
    } else if (layer == 1) {
        net->template layer_get<1>().init_layer(input_size, output_size);
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

void DenseDenseNet::storeWeights(const std::string& file) {
    net->store(file);
}

void DenseDenseNet::loadWeights(const std::string& file) {
    net->load(file);
}