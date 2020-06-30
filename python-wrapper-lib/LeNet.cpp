#include <typeinfo>
#include <memory>


#include "LeNet.h"

#include "dll/rbm/dyn_rbm.hpp"
#include "dll/rbm/dyn_conv_rbm.hpp"
#include "dll/transform/scale_layer.hpp"
#include "dll/transform/shape_3d_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/neural/dyn_conv_layer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

LeNet::LeNet() : net(std::make_unique<dbn_lenet>()) { }

void LeNet::display() {
    net->display();
}

void LeNet::displayPretty() {
    net->display_pretty();
}

void LeNet::setConvLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters,
                         size_t firstDimFilter, size_t secDimFilter) {
    if (layer == 0) {
        net->template layer_get<0>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    } else if (layer == 2) {
        net->template layer_get<2>().init_layer(inputChannels, firstDim, secDim, nbFilters, firstDimFilter,
                                                secDimFilter);
    }

}

void LeNet::setMPLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize,
                       size_t secDimPoolSize) {
    if (layer == 1) {
        net->template layer_get<1>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    } else if (layer == 3) {
        net->template layer_get<3>().init_layer(inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize);
    }
}

void LeNet::setDenseLayer(size_t layer, size_t inputSize, size_t outputSize) {
    if (layer == 4) {
        net->template layer_get<4>().init_layer(inputSize, outputSize);
    } else if (layer == 5) {
        net->template layer_get<5>().init_layer(inputSize, outputSize);
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


void all2() {
    typedef dll::dbn_desc<
            dll::dbn_layers <
            dll::dyn_conv_layer_desc < dll::activation < dll::function::TANH>>
    ::layer_t,
            dll::dyn_mp_2d_layer_desc < dll::weight_type < float >> ::layer_t,
            dll::dyn_conv_layer_desc < dll::activation < dll::function::TANH >> ::layer_t,
            dll::dyn_mp_2d_layer_desc < dll::weight_type < float >> ::layer_t,
            dll::dyn_dense_layer_desc < dll::activation < dll::function::RELU >> ::layer_t,
            dll::dyn_dense_layer_desc < dll::activation < dll::function::SOFTMAX >> ::layer_t >,
            dll::trainer < dll::sgd_trainer >, dll::updater < dll::updater_type::NADAM >, dll::batch_size <
                                                                                          100 >> ::dbn_t
    lenet;

    auto dbn_lenet = std::make_unique<lenet>();
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    /*
     * CNN
     * (input channels,
     *  first visible dim,
     *  sec visible dim,
     *  nb filters,
     *  first dim of filters,
     *  sec dim of filters)
     *
     *  kernel = filter
     */

    dbn_lenet->template layer_get<0>().init_layer(1, 28, 28, 6, 5, 5);
    dbn_lenet->template layer_get<1>().init_layer(6, 24, 24, 2, 2);
    dbn_lenet->template layer_get<2>().init_layer(6, 12, 12, 16, 5, 5);
    dbn_lenet->template layer_get<3>().init_layer(16, 8, 8, 2, 2);
    dbn_lenet->template layer_get<4>().init_layer(4 * 4 * 16, 150);
    dbn_lenet->template layer_get<5>().init_layer(150, 10);

    dbn_lenet->display();
    dbn_lenet->fine_tune(dataset.train(), 5);
    dbn_lenet->evaluate(dataset.test());
}


