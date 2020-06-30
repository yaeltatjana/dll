//
// Created by localuser on 29.06.20.
//

#include "AlexNet.h"


AlexNet::AlexNet() : net(std::make_unique<dbn_alexnet>()) {}

void AlexNet::display() {
    net->display();
}

void AlexNet::displayPretty() {
    net->display_pretty();
}

void AlexNet::setConvLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters,
                           size_t firstDimFilter, size_t secDimFilter) {
    if (layer == 0) {
        net->template layer_get<0>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 2) {
        net->template layer_get<2>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 4) {
        net->template layer_get<4>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 5) {
        net->template layer_get<5>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 6) {
        net->template layer_get<6>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    }

}

void AlexNet::setMPLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize,
                         size_t secDimPoolSize) {
    if (layer == 1) {
        net->template layer_get<1>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 3) {
        net->template layer_get<3>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 7) {
        net->template layer_get<7>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    }
}

void AlexNet::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    if (layer == 8) {
        net->template layer_get<8>().init_layer(inputSize, outputSize);
    } else if (layer == 9) {
        net->template layer_get<9>().init_layer(inputSize, outputSize);
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
