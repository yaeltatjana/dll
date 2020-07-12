#ifndef VGGNET16_H
#define VGGNET16_H

#include "dll/network.hpp"
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/neural/dyn_conv_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "../datasets/MnistReader.h"
#include "../datasets/TextReader.h"

using dbn_vggnet16 = dll::dbn_desc<
    dll::dbn_layers<
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
    dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

    dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
    dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
    dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,

    dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;



class VGGNet16 {
    std::unique_ptr <dbn_vggnet16> net;

public:
    VGGNet16();

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


#endif //VGGNET16_H
