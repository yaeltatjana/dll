#include "VGGNet19.h"
#include <memory>

VGGNet19::VGGNet19() : net(std::make_unique<dbn_vggnet19>()) { }

void VGGNet19::display() {
    net->display();
}

void VGGNet19::displayPretty() {
    net->display_pretty();
}

void VGGNet19::setConvLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t nbFilt, size_t filt1, size_t filt2) {

    switch(layer) {
        case 0:net->template layer_get<0>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 1:net->template layer_get<1>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 3:net->template layer_get<3>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 4:net->template layer_get<4>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 6:net->template layer_get<6>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 7:net->template layer_get<7>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 8:net->template layer_get<8>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 10:net->template layer_get<10>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 11:net->template layer_get<11>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 12:net->template layer_get<12>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 14:net->template layer_get<14>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 15:net->template layer_get<15>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
        case 16:net->template layer_get<16>().init_layer(channel, dim1, dim2, nbFilt, filt1, filt2); break;
    }

}

void VGGNet19::setMPLayer(size_t layer, size_t channel, size_t dim1, size_t dim2, size_t pool1, size_t pool2) {
    switch(layer) {
        case 2: net->template layer_get<2>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 5: net->template layer_get<5>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 9: net->template layer_get<9>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 13: net->template layer_get<13>().init_layer(channel, dim1, dim2, pool1, pool2); break;
        case 17: net->template layer_get<17>().init_layer(channel, dim1, dim2, pool1, pool2); break;
    }
}

void VGGNet19::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    switch(layer) {
        case 18: net->template layer_get<18>().init_layer(inputSize, outputSize); break;
        case 19: net->template layer_get<19>().init_layer(inputSize, outputSize); break;
        case 20: net->template layer_get<20>().init_layer(inputSize, outputSize); break;
    }
}

void VGGNet19::setLearningRate(double rate) {
    net->learning_rate = rate;
}

void VGGNet19::setAdamBeta1(double beta) {
    net->adam_beta1 = beta;
}

void VGGNet19::setAdamBeta2(double beta) {
    net->adam_beta2 = beta;
}

float VGGNet19::fineTune(MnistReader &ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void VGGNet19::evaluate(MnistReader &ds) {
    net->evaluate(ds.trainSet());
}

float VGGNet19::fineTune(TextReader &ds, size_t epochs) {
    return net->fine_tune(ds.getImages(),ds.readLabels(), epochs);
}

void VGGNet19::evaluate(TextReader &ds) {
    net->evaluate(ds.getImages(), ds.readLabels());
}

void VGGNet19::storeWeights(const std::string &file) {
    net->store(file);
}

void VGGNet19::loadWeights(const std::string &file) {
    net->load(file);
}