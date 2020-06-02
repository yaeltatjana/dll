#include <iostream>
#include "mnist_lib.h"

int main() {
    MnistLib lib;
    lib.createNet(28 * 28, 32, 0.001);
    lib.displayNet();
    lib.displayDataset();
    // lib.displayDatasetPretty();
    lib.train(5);
    lib.evaluate();

    // compare with simple example in one method
    // std::cout << "=================================================================" << std::endl;
    // doSimpleExample();

    return 0;
}