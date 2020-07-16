#ifndef MNISTREADER_H
#define MNISTREADER_H

#include "../dataset_types.h"

/**
 * Struct used to wrap the samples of the MNIST Dataset
 */
struct MnistDataset {
    std::vector <std::vector<uint8_t>> training_images;    ///< The training images
    std::vector <std::vector<uint8_t>> test_images;        ///< The test images
    std::vector <uint8_t> training_labels;                 ///< The training labels
    std::vector <uint8_t> test_labels;                     ///< The test labels

    /**
     * Constructor for the MNIST dataset
     * @param ds MNIST Dataset in form of the MNIST module
     */
    MnistDataset(mnist::MNIST_dataset <std::vector, std::vector<uint8_t>, uint8_t> ds);
};

/**
 * Class containing tools to read the MNIST Dataset
 */
class MnistReader {
    ds_mnist_t dataset; ///< MNIST dataset holder

public:
    /**
     * Constructor to create a reader
     */
    MnistReader();

    /**
     * Read the values contained in the dataset
     * @return Dataset split into train/test images and labels
     */
    MnistDataset readDataset();

    /**
     * Display the dataset
     */
    void display();

    /**
     * Display the dataset in a prettier way, ie in a table
     */
    void displayPretty();

    /**
     * Get the train set generator
     * @return Generator of MNIST train set
     */
    ds_trainG_t &trainSet();

    /**
     * Get the test set generator
     * @return Generator of MNIST test set
     */
    ds_testG_t &testSet();
};


#endif //MNISTREADER_H
