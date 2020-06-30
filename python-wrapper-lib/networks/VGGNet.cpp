#include "VGGNet.h"

#include <memory>
#include <array>

VGGNet::VGGNet() : net(std::make_unique<dbn_vggnet>()) { }

void VGGNet::display() {
    net->display();
}

void VGGNet::displayPretty() {
    net->display_pretty();
}

void VGGNet::setConvLayer(const size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters,
                         size_t firstDimFilter, size_t secDimFilter) {

    if (layer == 0) {
        net->template layer_get<0>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 1) {
        net->template layer_get<1>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 3) {
        net->template layer_get<3>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 4) {
        net->template layer_get<4>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 6) {
        net->template layer_get<6>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 7) {
        net->template layer_get<7>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 8) {
        net->template layer_get<8>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 10) {
        net->template layer_get<10>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 11) {
        net->template layer_get<11>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 12) {
        net->template layer_get<12>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 14) {
        net->template layer_get<14>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 15) {
        net->template layer_get<15>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 16) {
        net->template layer_get<16>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    }

}

void VGGNet::setMPLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize,
                       size_t secDimPoolSize) {
    if (layer == 2) {
        net->template layer_get<2>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 5) {
        net->template layer_get<5>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 9) {
        net->template layer_get<9>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 13) {
        net->template layer_get<13>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 17) {
        net->template layer_get<17>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    }
}

void VGGNet::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    if (layer == 18) {
        net->template layer_get<18>().init_layer(inputSize, outputSize);
    } else if (layer == 19) {
        net->template layer_get<19>().init_layer(inputSize, outputSize);
    } else if (layer == 20) {
        net->template layer_get<20>().init_layer(inputSize, outputSize);
    }
}

void VGGNet::setLearningRate(double rate) {
    net->learning_rate = rate;
}

void VGGNet::setAdamBeta1(double beta) {
    net->adam_beta1 = beta;
}

void VGGNet::setAdamBeta2(double beta) {
    net->adam_beta2 = beta;
}

float VGGNet::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void VGGNet::evaluate(MnistReader &ds) {
    net->evaluate(ds.trainSet());
}

void VGGNet::storeWeights(const std::string &file) {
    net->store(file);
}

void VGGNet::loadWeights(const std::string &file) {
    net->load(file);
}