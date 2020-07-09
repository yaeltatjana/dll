#include "Mnist3DReader.h"

Mnist3DReader::Mnist3DReader() :
        dataset(mnist::read_dataset_3d < std::vector, etl::dyn_matrix < float, 3 >> ()) {}

std::vector <std::vector<uint8_t>> &Mnist3DReader::readTrainingImages() {
    static std::vector <std::vector<uint8_t>> imgs = mnist::read_dataset().training_images;
    return imgs;
}

std::vector <std::vector<uint8_t>> &Mnist3DReader::readTestImages() {
    static std::vector <std::vector<uint8_t>> imgs = mnist::read_dataset().test_images;
    return imgs;
}

std::vector <uint8_t> &Mnist3DReader::readTrainingLabels() {
    return dataset.training_labels;
}

std::vector <uint8_t> &Mnist3DReader::readTestLabels() {
    return dataset.test_labels;
}

std::vector <etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> &Mnist3DReader::getTrainingImages() {
    return dataset.training_images;
}

std::vector <etl::dyn_matrix_impl<float, etl::order::RowMajor, 3>> &Mnist3DReader::getTestImages() {
    return dataset.test_images;
}