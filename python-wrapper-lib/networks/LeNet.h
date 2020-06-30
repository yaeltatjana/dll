#ifndef LENET_H
#define LENET_H

#include <vector>
#include "../network_types.h"
#include "../datasets/Mnist3DReader.h"
#include "../datasets/MnistReader.h"


class LeNet {
    std::unique_ptr <dbn_lenet> net;

public:
    /**
     * Create a instance without initializing the layers
     */
    LeNet();

    /**
     * Method to display the network
     */
    void display();

    /**
     * Method to display the network in the pretty form
     */
    void displayPretty();

    void setConvLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters,
                      size_t firstDimFilter, size_t secDimFilter);

    void setMPLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize,
                    size_t secDimPoolSize);

    void setDenseLayer(size_t layer, size_t inputSize, size_t outputSize);

    void setLearningRate(double rate);

    void setAdamBeta1(double beta);

    void setAdamBeta2(double beta);

    /**
     * Method to train the network with a given dataset
     * @param ds MnistDataset object which contains the dataset
     * @param epochs Number of epochs
     * @return Final classification error
     */
    float fineTune(MnistReader &ds, size_t epochs);

    /**
     * Method to evaluate the network
     * @param ds MnistDataset object which contains the dataset
     */
    void evaluate(MnistReader &ds);

    /**
     * Method to store the weights in a file
     * @param file Name of the file where to save
     */
    void storeWeights(const std::string &file);

    /**
     * Method to load the weights from a file
     * @param file Name of the file where load the weights
     */
    void loadWeights(const std::string &file);


};

void all2();


#endif //LENET_H
