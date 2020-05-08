#ifndef DLL_MNIST_LIB_H
#define DLL_MNIST_LIB_H

#include <vector>

/**
 * Structure containing the MNIST dataset
 */
struct Dataset {
    std::vector <std::vector<uint8_t>> training_images;
    std::vector <std::vector<uint8_t>> test_images;
    std::vector <uint8_t> training_labels;
    std::vector <uint8_t> test_labels;
};

/**
 * Class used to create a neural network and to train/evaluate it on MNIST dataset
 * It actually only allows to get the dataset
 */
class MnistLib {
private:
    struct Dataset ds;

public:
    MnistLib();

    struct Dataset getDataset();
};

/**
 * Function to launch the simple example
 */
void doSimpleExample();

#endif //DLL_MNIST_LIB_H
