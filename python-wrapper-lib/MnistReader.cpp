#include "MnistReader.h"
#include "dll/datasets.hpp"
#include <typeinfo>
#include <iostream>


MnistReader::MnistReader() : dataset(dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {})) {}

mnist::MNIST_dataset<std::vector, std::vector<uint8_t>,uint8_t> MnistReader::readDataset() {
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>,uint8_t> ds = mnist::read_dataset();
    return ds;
}

void MnistReader::display() {
    dataset.display();
}

void MnistReader::displayPretty() {
    dataset.display_pretty();
}

ds_trainG_t& MnistReader::trainSet() {
    // std::cout << typeid(dataset.train()).name() << std::endl;
    return dataset.train();
}

/*void MnistReader::validationSet() {
    dataset.val();
}*/

ds_testG_t& MnistReader::testSet() {
    return dataset.test();
}