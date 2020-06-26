#ifndef DENSEDENSEDENSENET_H
#define DENSEDENSEDENSENET_H

#include <memory>
#include "network_types.h"
#include "MnistReader.h"

/**
 * Network with 3 Dense layers : relu -> relu -> softmax
 * Updater (optimization algo) : Stochastic Gradient Descent with MOMENTUM
 * Loss function: CATEGORICAL_CROSS_ENTROPY
 * Batch size: 100
 */
class DenseDenseDenseNet {
    std::unique_ptr <dbn_3dense_RRSo> net;

public:
    /**
     * Default constructor with no initialization of the layers
     */
    DenseDenseDenseNet();

    /**
     * Constructor with input/output sizes
     * @param nb_input Vector containing sizes of layers
     * @param nb_output Vector containing sizes of the layers
     */
    DenseDenseDenseNet(std::vector <size_t> &nb_input, std::vector <size_t> &nb_output);

    /**
     * Method to change the learning rate value, default to 0.1
     * @param l_rate Value of learning rate
     */
    void setLearningRate(double l_rate);

    /**
     * Method to set/change the value of input and output sizes of a layer
     * @param layer Index of the layer
     * @param input_size  New input size
     * @param output_size New output size
     */
    void setLayerSize(size_t layer, size_t input_size, size_t output_size);

    /**
     * Method to change the initial momentum, default to 0.9
     * @param m
     */
    void setInitialMomentum(double m);

    /**
     * Method to display the network
     */
    void display();

    /**
     * Method to display the network in the pretty form
     */
    void displayPretty();

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


#endif //DENSEDENSEDENSENET_H
