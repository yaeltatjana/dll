#include <vector>
#include <memory>
#include <utility>

#include "dll/neural/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"
#include "mnist_lib.h"

MnistLib::MnistLib() {
    mnist::MNIST_dataset mnist_dataset = mnist::read_dataset();
    ds = {
        mnist_dataset.training_images,
        mnist_dataset.test_images,
        mnist_dataset.training_labels,
        mnist_dataset.test_labels,
    };
}


MnistLib::~MnistLib(){
    std::cout << "Destructor MnistLib" << std::endl;
}

struct Dataset MnistLib::getDataset() {
    return ds;
}

mytype& MnistLib::createNN() {
    net = std::make_unique<mytype>();
    return *net;
}

void MnistLib::displayNN(std::unique_ptr<mytype>& nn){
    nn->display();
}

void MnistLib::displayDS(){
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    dataset.display();
}

float MnistLib::train(std::unique_ptr<mytype>& nn){
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    return nn->fine_tune(dataset.train(), 5);
}

void MnistLib::evaluate(std::unique_ptr<mytype>& nn){
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    nn->evaluate(dataset.test());
}


void doSimpleExample() {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    // Build the network
    using network_t = dll::dyn_network_desc <
    dll::network_layers<dll::dense_layer < 28 * 28, 32, dll::relu>>,
    dll::batch_size < 100 >,
            dll::shuffle,       // shuffle before each epoch
            dll::updater < dll::updater_type::RMSPROP >
            > ::network_t;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network with 5 epochs
    net->fine_tune(dataset.train(), 5);

    // Test the network on test set
    net->evaluate(dataset.test());
}