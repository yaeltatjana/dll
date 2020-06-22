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

/*void DenseDenseNet::setRmspropDecay(double d) {
    net->rmsprop_decay = d;
}*/

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

void DenseDenseNet::all() {
    // initialization
    using dbn =
    dll::dbn_desc<
            dll::dbn_layers < dll::dyn_dense_layer_desc < dll::activation < dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc < dll::activation < dll::function::SOFTMAX >> ::layer_t >,
            dll::updater < dll::updater_type::MOMENTUM >,
            dll::trainer < dll::sgd_trainer >,
            dll::shuffle,
            dll::batch_size < 100 >> ::dbn_t;

    auto n = std::make_unique<dbn>();
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    n->template layer_get<0>().init_layer(28 * 28, 28 * 28);
    n->template layer_get<1>().init_layer(28 * 28, 32);
    n->learning_rate = 0.001;

    // display infos
    ds.display();
    n->display();

    // train
    n->fine_tune(ds.train(), 5);

    // evaluate
    n->evaluate(ds.test());
}