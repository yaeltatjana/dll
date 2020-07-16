#ifndef TEXTREADER_H
#define TEXTREADER_H

#include <vector>
#include "dll/datasets.hpp"

/**
 * Class that contains a dataset from a text file
 */
class TextReader {
    std::vector <std::vector<uint8_t>> images;                      ///< The images for reading
    std::vector <etl::fast_dyn_matrix<float, 1, 28, 28>> imgs;      ///< The images for training/evaluating
    std::vector <uint8_t> labels;                                   ///< The labels

public:
    /**
     * Constructor for a text reader
     * @param imgsPath      path to images' file
     * @param labelsPath    path to labels' file
     * @param imgLimit      limit size of images
     * @param labelLimit    limit size of labels
     */
    TextReader(std::string imgsPath, std::string labelsPath, size_t imgLimit = 0, size_t labelLimit = 0);

    /**
     * Read images of dataset
     * @return matrix of images
     */
    std::vector <std::vector<uint8_t>> &readImages();

    /**
     * Read images of dataset, especially for the train and evaluate methods
     * @return structure containing the images
     */
    std::vector <etl::fast_dyn_matrix<float, 1, 28, 28>> &getImages();

    /**
     * Read labels of dataset
     * @return
     */
    std::vector <uint8_t> &readLabels();
};


#endif //TEXTREADER_H
