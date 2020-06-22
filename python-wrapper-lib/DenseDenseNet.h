#ifndef DENSEDENSENET_H
#define DENSEDENSENET_H


#include <vector>
#include "network_types.h"
#include "MnistReader.h"



class DenseDenseNet {
    std::unique_ptr<dbn_dense_RSo> net;

public:
    DenseDenseNet();

    DenseDenseNet(std::vector<size_t>& nb_input, std::vector<size_t>& nb_output, double learning_rate);

    void display();

    float fineTune(MnistReader& ds, size_t epochs);

    void evaluate(MnistReader& ds);

    void all();
};


#endif //DENSEDENSENET_H
