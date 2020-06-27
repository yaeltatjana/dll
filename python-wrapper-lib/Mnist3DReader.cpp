#include "Mnist3DReader.h"
#include <typeinfo>

Mnist3DReader::Mnist3DReader() :
        dataset(mnist::read_dataset_3d < std::vector, etl::dyn_matrix < float, 3 >> (1000)) {}

std::vector<etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> Mnist3DReader::getTrainingImages() {
    return dataset.training_images;
}

std::vector<etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> Mnist3DReader::getTestImages() {
    return dataset.test_images;
}

std::vector <uint8_t> Mnist3DReader::getTrainingLabels() {
    return dataset.training_labels;
}

std::vector <uint8_t> Mnist3DReader::getTestLabels() {
    return dataset.test_labels;
}