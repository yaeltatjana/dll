#include <iostream>
#include "mnist_lib.h"

int main() {
    MnistLib lib;

    std::unique_ptr<mytype> nn;
    nn.reset(&lib.createNN());
    lib.displayDS();
    lib.displayNN(nn);
    lib.train(nn);

    // doSimpleExample();

    return 0;
}