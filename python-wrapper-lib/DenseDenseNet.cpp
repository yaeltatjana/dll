#include <memory>
#include "DenseDenseNet.h"

DenseDenseNet::DenseDenseNet() : net(std::make_unique<dbn_dense_RSo>()) { }

DenseDenseNet::DenseDenseNet(std::vector<size_t>& nb_input, std::vector<size_t>& nb_output, double l_rate) :
    net(std::make_unique<dbn_dense_RSo>()) {
    net->template layer_get<0>().init_layer(nb_input[0], nb_output[0]);
    net->template layer_get<1>().init_layer(nb_input[1], nb_output[1]);
    net->learning_rate = l_rate;
}


void DenseDenseNet::display() {
    net->display();
}

float DenseDenseNet::fineTune(MnistReader& ds, size_t epochs) {
    return net->fine_tune(ds.trainSet(), epochs);
}

void DenseDenseNet::evaluate(MnistReader& ds) {
    net->evaluate(ds.testSet());
}

void DenseDenseNet::all() {
    using dbn =
    dll::dbn_desc<
            dll::dbn_layers < dll::dyn_dense_layer_desc <dll::activation <dll::function::RELU>>::layer_t,
            dll::dyn_dense_layer_desc <dll::activation <dll::function::SOFTMAX>>::layer_t>,
    dll::updater <dll::updater_type::RMSPROP>,
            dll::trainer <dll::sgd_trainer>,
            dll::shuffle,
            dll::batch_size<100>>::dbn_t;

    auto n = std::make_unique<dbn>();
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    std::vector<size_t> in {28 * 28,28 * 28};
    std::vector<size_t> out {32,32};
    n->template layer_get<0>().init_layer(in[0], in[0]);
    n->template layer_get<1>().init_layer(out[1], out[1]);
    net->learning_rate = 0.001;

    n->fine_tune(ds.train(), 5);

}