#ifndef TEXTREADER_H
#define TEXTREADER_H

#include <vector>
#include "dll/datasets.hpp"


class TextReader {
    std::vector <std::vector<uint8_t>> images;
    std::vector<etl::fast_dyn_matrix<float, 1, 28, 28>> imgs;
    std::vector <uint8_t> labels;

public:
    TextReader(std::string imgsPath, std::string labelsPath, size_t imgLimit = 0, size_t labelLimit = 0);

    std::vector <std::vector<uint8_t>>& readImages();

    std::vector<etl::fast_dyn_matrix<float, 1, 28, 28>>& getImages();

    std::vector <uint8_t>& readLabels();
};


#endif //TEXTREADER_H
