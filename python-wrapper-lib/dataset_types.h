#ifndef DATASET_TYPES_H
#define DATASET_TYPES_H

#include "dll/datasets.hpp"
/**
 * Type of the dataset holder
 */
using ds_mnist_t =
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


/**
 * Type of the train generator
 */
using ds_trainG_t = dll::inmemory_data_generator<
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
void>;

/**
 * Type of the test generator
 */
using ds_testG_t = dll::inmemory_data_generator<
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
void>;

#endif //DATASET_TYPES_H
