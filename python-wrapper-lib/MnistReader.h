#ifndef MNISTREADER_H
#define MNISTREADER_H

#include "dataset_types.h"

class MnistReader {
    ds_mnist_t dataset;

public:
    MnistReader();
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>,uint8_t> readDataset();
    void display();
    void displayPretty();
    ds_trainG_t& trainSet();
    //void validationSet();
    ds_testG_t& testSet();
};


#endif //MNISTREADER_H
