#include <iostream>
#include <string>

#include "TextReader.h"
#include "dll/text_reader.hpp"

TextReader::TextReader(std::string imgsPath, std::string labelsPath, size_t imgLimit, size_t labelLimit) :
        images(dll::text::read_images < std::vector, std::vector < uint8_t > , false > (imgsPath, imgLimit)),
        labels(dll::text::read_labels<std::vector, uint8_t>(labelsPath, labelLimit)) {}

std::vector <std::vector<uint8_t>> TextReader::readImages() {
    return images;
}

std::vector <uint8_t> TextReader::readLabels() {
    return labels;
}