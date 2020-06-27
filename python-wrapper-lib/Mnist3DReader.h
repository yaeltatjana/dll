#ifndef MNIST3DREADER_H
#define MNIST3DREADER_H

#include <vector>

#include "dataset_types.h"
#include "dll/datasets.hpp"

class Mnist3DReader {
    ds_mnist_3d_t dataset;

public:
    Mnist3DReader();

    std::vector<etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> getTrainingImages();

    std::vector<etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> getTestImages();

    std::vector <uint8_t> getTrainingLabels();

    std::vector <uint8_t> getTestLabels();
};


#endif //MNIST3DREADER_H
