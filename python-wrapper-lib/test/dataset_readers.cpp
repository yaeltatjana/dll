#include <vector>
#include <string>
#include "dll/datasets.hpp"
#include "dll/text_reader.hpp"


void test_mnist() {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});
    ds.display();
    ds.display_pretty();
}

void test_text() {
    // nothing interesting
    auto imgs = dll::text::read_images < std::vector, std::vector < uint8_t > , false > ("test/text_db/images", 0);
    auto lbls = dll::text::read_labels<std::vector, uint8_t>("test/text_db/labels", 0);
}

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::string test = argv[i];
        if(test.compare("mnist") == 0) test_mnist();
        if(test.compare("text") == 0) test_text();
    }

    return 0;
}