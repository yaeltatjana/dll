#ifndef DLL_MNIST_LIB_H
#define DLL_MNIST_LIB_H

#include <vector>
#include <memory>
#include "dll/neural/dense_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

/**
 * Structure containing the MNIST dataset
 */
struct Dataset {
    std::vector <std::vector<uint8_t>> training_images;
    std::vector <std::vector<uint8_t>> test_images;
    std::vector <uint8_t> training_labels;
    std::vector <uint8_t> test_labels;
};

typedef dll::dbn<
    dll::generic_dyn_dbn_desc<
        dll::dbn,
        dll::network_layers<
            dll::dense_layer<28 * 28,32,dll::relu>
        >,
        dll::batch_size<100>,
        dll::shuffle,
        dll::updater<dll::updater_type::RMSPROP>
    >
> mytype;


/**
 * Class used to create a neural network and to train/evaluate it on MNIST dataset
 * It actually only allows to get the dataset
 */
class MnistLib {
private:
    struct Dataset ds;
    std::unique_ptr<mytype> net;

public:
    MnistLib();

    ~MnistLib();

    struct Dataset getDataset();

    mytype& createNN();

    void displayNN(std::unique_ptr<mytype>& nn);

    void displayDS();

    float train(std::unique_ptr<mytype>& nn);

    void evaluate(std::unique_ptr<mytype>& nn);
};

/**
 * Function to launch the simple example
 */
void doSimpleExample();

#endif //DLL_MNIST_LIB_H
