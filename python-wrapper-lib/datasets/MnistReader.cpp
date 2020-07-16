#include "MnistReader.h"
#include "dll/datasets.hpp"
#include <iostream>

MnistDataset::MnistDataset(mnist::MNIST_dataset <std::vector, std::vector<uint8_t>, uint8_t> ds) {
    training_images = ds.training_images;
    test_images = ds.test_images;
    training_labels = ds.training_labels;
    test_labels = ds.test_labels;
}

MnistReader::MnistReader() : dataset(dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {})) {}

MnistDataset MnistReader::readDataset() {
    static MnistDataset ds(mnist::read_dataset());
    return mnist::read_dataset();
}

void MnistReader::display() {
    dataset.display();
}

void MnistReader::displayPretty() {
    dataset.display_pretty();
}

ds_trainG_t &MnistReader::trainSet() {
    return dataset.train();
}

ds_testG_t &MnistReader::testSet() {
    return dataset.test();
}