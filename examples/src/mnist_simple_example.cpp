//
// Created by yael
//

#include "dll/neural/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"
#include <typeinfo>
#include <vector>

// simple example of DLL use
int main(int /*argc*/, char * /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    // Build the network
    using network_t = dll::dyn_network_desc <
        dll::network_layers<dll::dense_layer < 28 * 28, 32, dll::relu>>,
        dll::batch_size < 100 >,
        dll::shuffle,       // shuffle before each epoch
        dll::updater < dll::updater_type::RMSPROP >
    > ::network_t;

    // print type of network with RTTI -> remove flag -fno-rtti in Makefile
    // std::cout << Typeid(network_t).name() << std::endl;

    auto net = std::make_unique<network_t>();

    // Display the network and dataset
    net->display();
    dataset.display();

    // Train the network with 5 epochs
    net->fine_tune(dataset.train(), 5);

    // Test the network on test set
    net->evaluate(dataset.test());

    return 0;
}

