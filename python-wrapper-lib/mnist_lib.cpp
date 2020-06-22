#include <vector>
#include <memory>
#include <utility>
#include <typeinfo>

#include "dll/neural/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"
#include "mnist_lib.h"
#include "mnist_lib_types.h"

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

void MnistLib::createNet(size_t nb_input, size_t nb_output, double learning_rate) {
    net_dbn = std::make_unique<dbn_t>();
    net_dbn->template layer_get<0>().init_layer(nb_input, nb_output);
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

float MnistLib::train(size_t epochs) {
    return net_dbn->fine_tune(dataset.train(), epochs);
}

void MnistLib::evaluate() {
    net_dbn->evaluate(dataset.test());
}

void doSimpleExample() {
    // Load the dataset
    auto dataset1 = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

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

ds_t getMNISTDataset() {
    return dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
}

// dense layer with relu activation
std::unique_ptr<dbn_relu>& createDenseRelu(size_t nb_input, size_t nb_output, double learning_rate) {
    static std::unique_ptr<dbn_relu> net = std::make_unique<dbn_relu>();
    net->template layer_get<0>().init_layer(nb_input, nb_output);
    net->learning_rate = learning_rate;
    return net;
}

void displayDenseRelu(std::unique_ptr<dbn_relu>& net) {
    net->display();
}

float trainDenseRelu(std::unique_ptr<dbn_relu>& net, size_t epochs) {
    ds_t ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    return net->fine_tune(ds.train(), epochs);
}

// 3x dense layers : relu -> relu -> softmax
std::unique_ptr<dbn_RRSo>& createDenseRRSo(std::list<size_t> nb_input, std::list<size_t> nb_output, size_t learning_rate) {
    static std::unique_ptr<dbn_RRSo> net = std::make_unique<dbn_RRSo>();

    net->template layer_get<0>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();

    net->template layer_get<1>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();

    net->template layer_get<2>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();

    net->learning_rate = learning_rate;
    return net;
}

void displayDenseRRSo(std::unique_ptr<dbn_RRSo>& net) {
    net->display();
}

float trainDenseRRSo(std::unique_ptr<dbn_RRSo>& net, size_t epochs) {
    ds_t ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    return net->fine_tune(ds.train(), epochs);
}

void allRRSo() {
    static std::unique_ptr<dbn_RRSo> net = std::make_unique<dbn_RRSo>();
    std::list <size_t> nb_input = {28 * 28, 28 * 28, 28 * 28};
    std::list <size_t> nb_output = {32, 32, 32};

    net->template layer_get<0>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();
    net->template layer_get<1>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();
    net->template layer_get<2>().init_layer(nb_input.front(), nb_output.front());
    nb_input.pop_front(); nb_output.pop_front();
    net->learning_rate = 0.001;

    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    net->display();
    ds.display();

    // Train the network with 5 epochs
    net->fine_tune(ds.train(), 5);

    // Test the network on test set
    net->evaluate(ds.test());
}

