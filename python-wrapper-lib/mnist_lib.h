#ifndef DLL_MNIST_LIB_H
#define DLL_MNIST_LIB_H

#include <vector>
#include <memory>
#include "dll/neural/dense_layer.hpp"
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/dbn.hpp"
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

/**
 * Type used for the neural network for the simple example
 */
using dbn_t =
dll::dbn_desc<
        dll::dbn_layers <
        dll::dyn_dense_layer_desc < dll::activation < dll::function::RELU>>::layer_t>,
dll::updater <dll::updater_type::RMSPROP>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;

constexpr std::size_t alloc = 4;

/**
 * Type of the dataset holder
 */
using ds_t =
dll::dataset_holder<
        dll::inmemory_data_generator<
                const etl::fast_matrix_impl<
                        float,
                        std::vector<float, cpp::aligned_allocator < float, alloc> >,
                etl::order::RowMajor,
                1,
                28,
                28> * ,
        const float *,
        dll::inmemory_data_generator_desc <
        dll::batch_size < 100>,
dll::scale_pre<255>,
dll::categorical>,
void>,
dll::inmemory_data_generator<
        const etl::fast_matrix_impl<
                float,
                std::vector<float, cpp::aligned_allocator < float, alloc> >,
        etl::order::RowMajor,
        1,
        28,
        28>*,
const float*,
dll::inmemory_data_generator_desc <
dll::batch_size<100>,
dll::scale_pre<255>,
dll::categorical>,
void>,
int>;

/**
 * Class used to create a neural network and to train/evaluate it on MNIST dataset
 * It actually only allows to get the dataset
 */
class MnistLib {
private:
    struct Dataset ds;
    std::unique_ptr <dbn_t> net_dbn;
    ds_t dataset = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

public:
    MnistLib();

    struct Dataset getDataset();

    void createNet(int nb_visibles, int nb_hiddens, int learning_rate);

    void displayDataset();

    void displayDatasetPretty();

    void displayNet();

    float train(int epochs);

    void evaluate();
};

/**
 * Function to launch the simple example
 */
void doSimpleExample();

#endif //DLL_MNIST_LIB_H
