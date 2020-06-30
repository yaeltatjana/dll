#ifndef MNIST3DREADER_H
#define MNIST3DREADER_H

#include <vector>

#include "../dataset_types.h"
#include "dll/datasets.hpp"

class Mnist3DReader {
    ds_mnist_3d_t dataset;

public:
    Mnist3DReader();

    std::vector <std::vector<uint8_t>> &readTrainingImages();

    std::vector <std::vector<uint8_t>> &readTestImages();

    std::vector <uint8_t> &readTrainingLabels();

    std::vector <uint8_t> &readTestLabels();

    std::vector <etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> &getTrainingImages();

    std::vector <etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> &getTestImages();

};


#endif //MNIST3DREADER_H
