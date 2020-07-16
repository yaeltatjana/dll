#ifndef DENSEDENSEDENSENET_H
#define DENSEDENSEDENSENET_H

#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/network.hpp"
#include "../datasets/MnistReader.h"
#include "../datasets/TextReader.h"

#include <memory>

/**
 * Type of network
 */
using dbn_3dense = dll::dbn_desc<
        dll::dbn_layers <
        dll::dyn_dense_layer_desc < dll::activation < dll::function::RELU>>::layer_t,
dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>
::layer_t,
dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>
::layer_t>,
dll::updater <dll::updater_type::MOMENTUM>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;

/**
 * Network with 3 Dense layers : relu -> relu -> softmax
 * Updater (optimization algo) : Stochastic Gradient Descent with MOMENTUM
 * Loss function: CATEGORICAL_CROSS_ENTROPY
 * Batch size: 100
 */
class DenseDenseDenseNet {
    std::unique_ptr <dbn_3dense> net;   ///< The network

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
     * Method to train the network with a given dataset
     * @param ds TextReader object which contains the dataset
     * @param epochs Number of epochs
     * @return Final classification error
     */
    float fineTune(TextReader &ds, size_t epochs);

    /**
     * Method to evaluate the network
     * @param ds TextReader object which contains the dataset
     */
    void evaluate(TextReader &ds);

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
