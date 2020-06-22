#ifndef KERAS4DLL_II_TYPES_H
#define KERAS4DLL_II_TYPES_H



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

using dense_relu = dll::dyn_dense_layer_desc <dll::activation <dll::function::RELU>>::layer_t;
using dense_sig = dll::dyn_dense_layer_desc <dll::activation <dll::function::SIGMOID>>::layer_t;
using dense_soft = dll::dyn_dense_layer_desc <dll::activation <dll::function::SOFTMAX>>::layer_t;
using dense_id = dll::dyn_dense_layer_desc <dll::activation <dll::function::IDENTITY>>::layer_t;
using dense_tanh = dll::dyn_dense_layer_desc <dll::activation <dll::function::TANH>>::layer_t;

using dbn_relu =
dll::dbn_desc<
        dll::dbn_layers <dense_relu>,
dll::updater <dll::updater_type::RMSPROP>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;

using dbn_sig =
dll::dbn_desc<
        dll::dbn_layers <dense_sig>,
dll::updater <dll::updater_type::RMSPROP>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;

using dbn_RRSo =
dll::dbn_desc<
        dll::dbn_layers <dense_relu,dense_relu,dense_soft>,
dll::updater <dll::updater_type::RMSPROP>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;


/**
 * Type of the dataset holder
 */

using ds_t =
dll::dataset_holder<
        dll::inmemory_data_generator<
                const etl::fast_matrix_impl<
                        float,
                        std::vector<float, cpp::aligned_allocator < float, etl::default_intrinsic_traits<float>::alignment>>,
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
                std::vector<float, cpp::aligned_allocator < float, etl::default_intrinsic_traits<float>::alignment >>,
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

#endif //KERAS4DLL_II_TYPES_H
