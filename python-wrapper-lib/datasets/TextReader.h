#ifndef TEXTREADER_H
#define TEXTREADER_H


class TextReader {
    std::vector <std::vector<uint8_t>> images;
    std::vector <uint8_t> labels;

public:
    TextReader(std::string imgsPath, std::string labelsPath, size_t imgLimit = 0, size_t labelLimit = 0);

    std::vector <std::vector<uint8_t>> readImages();

    std::vector <uint8_t> readLabels();
};


#endif //TEXTREADER_H
