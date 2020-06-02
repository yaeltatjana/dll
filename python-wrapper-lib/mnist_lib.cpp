#include <vector>
#include <memory>
#include <utility>
#include <typeinfo>

#include "dll/neural/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"
#include "mnist_lib.h"


MnistLib::MnistLib() : dataset(dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {})) {
    mnist::MNIST_dataset mnist_dataset = mnist::read_dataset();
    ds = {
            mnist_dataset.training_images,
            mnist_dataset.test_images,
            mnist_dataset.training_labels,
            mnist_dataset.test_labels,
    };
}

struct Dataset MnistLib::getDataset() {
    return ds;
}

void MnistLib::createNet(int nb_visibles, int nb_hiddens, int learning_rate) {
    net_dbn = std::make_unique<dbn_t>();
    net_dbn->template layer_get<0>().init_layer(nb_visibles, nb_hiddens);
    net_dbn->learning_rate = learning_rate;
}

void MnistLib::displayDataset() {
    dataset.display();
}

void MnistLib::displayDatasetPretty() {
    dataset.display_pretty();
}

void MnistLib::displayNet() {
    net_dbn->display();
}

float MnistLib::train(int epochs) {
    return net_dbn->fine_tune(dataset.train(), epochs);
}

void MnistLib::evaluate() {
    net_dbn->evaluate(dataset.test());
}

void doSimpleExample() {
    // Load the dataset
    auto dataset1 = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    //std::cout << typeid(dataset1).name() << std::endl;

    // Build the network
    using network_t = dll::dyn_network_desc <
    dll::network_layers<dll::dense_layer < 28 * 28, 32, dll::relu>>,
    dll::batch_size < 100 > ,
            dll::shuffle,       // shuffle before each epoch
            dll::updater < dll::updater_type::RMSPROP > > ::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display();
    dataset1.display();

    // Train the network with 5 epochs
    net->fine_tune(dataset1.train(), 5);

    // Test the network on test set
    net->evaluate(dataset1.test());
}