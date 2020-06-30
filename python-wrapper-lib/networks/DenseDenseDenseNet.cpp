#include "DenseDenseDenseNet.h"

#include <memory>

DenseDenseDenseNet::DenseDenseDenseNet() : net(std::make_unique<dbn_3dense_RRSo>()) {}

DenseDenseDenseNet::DenseDenseDenseNet(std::vector <size_t> &nb_input, std::vector <size_t> &nb_output) :
        net(std::make_unique<dbn_3dense_RRSo>()) {
    net->template layer_get<0>().init_layer(nb_input[0], nb_output[0]);
    net->template layer_get<1>().init_layer(nb_input[1], nb_output[1]);
    net->template layer_get<2>().init_layer(nb_input[1], nb_output[2]);
}

void DenseDenseDenseNet::setLearningRate(double l_rate) {
    net->learning_rate = l_rate;
}

void DenseDenseDenseNet::setLayerSize(size_t layer, size_t input_size, size_t output_size) {
    if (layer == 0) {
        net->template layer_get<0>().init_layer(input_size, output_size);
    } else if (layer == 1) {
        net->template layer_get<1>().init_layer(input_size, output_size);
    } else if (layer == 2) {
        net->template layer_get<2>().init_layer(input_size, output_size);
    }
}

void DenseDenseDenseNet::setInitialMomentum(double m) {
    net->initial_momentum = m;
}

/*void DenseDenseDenseNet::setRmspropDecay(double d) {
    net->rmsprop_decay = d;
}*/

void DenseDenseDenseNet::display() {
    net->display();
}

void DenseDenseDenseNet::displayPretty() {
    net->display_pretty();
}

float DenseDenseDenseNet::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void DenseDenseDenseNet::evaluate(MnistReader &ds) {
    net->evaluate(ds.testSet());
}
    // void setRmspropDecay(double d);

void DenseDenseDenseNet::storeWeights(const std::string& file) {
    net->store(file);
}

void DenseDenseDenseNet::loadWeights(const std::string& file) {
    net->load(file);
}