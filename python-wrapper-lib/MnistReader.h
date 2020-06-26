#ifndef MNISTREADER_H
#define MNISTREADER_H

#include "dataset_types.h"
#include "dll/datasets.hpp"

class MnistReader {
    ds_mnist_t dataset;

public:
    MnistReader();
    MnistDataset& readDataset();
    void display();
    void displayPretty();
    ds_trainG_t& trainSet();
    //void validationSet();
    ds_testG_t& testSet();
};


#endif //MNISTREADER_H
