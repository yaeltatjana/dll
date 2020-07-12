#include <string>
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/neural/dyn_conv_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "dll/datasets.hpp"
#include "dll/network.hpp"

void test_dd(size_t epochs) {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    using dd = dll::dbn_desc<dll::dbn_layers <
    dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
    dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
    dll::updater <dll::updater_type::MOMENTUM>,
    dll::trainer <dll::sgd_trainer>,
    dll::shuffle,
    dll::batch_size<100>>
    ::dbn_t;

    auto net = std::make_unique<dd>();
    net->template layer_get<0>().init_layer(28*28, 16);
    net->template layer_get<1>().init_layer(16, 10);
    net->initial_momentum = 0.85;

    net->display();
    net->display_pretty();
    net->fine_tune(ds.train(), epochs);
    net->evaluate(ds.test());
}


void test_ddd(size_t epochs) {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    using dbn = dll::dbn_desc<
        dll::dbn_layers <
        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
        dll::updater <dll::updater_type::MOMENTUM>,
        dll::trainer <dll::sgd_trainer>,
        dll::shuffle,
        dll::batch_size<100>>
        ::dbn_t;
    auto net = std::make_unique<dbn>();
    net->template layer_get<0>().init_layer(28 * 28, 32);
    net->template layer_get<1>().init_layer(32, 16);
    net->template layer_get<2>().init_layer(16, 10);
    net->initial_momentum = 0.85;

    net->display();
    net->display_pretty();
    net->fine_tune(ds.train(), epochs);
    net->evaluate(ds.test());
}


void test_lenet(size_t epochs) {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
        dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::TANH>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;


    auto net = std::make_unique<dbn_t>();
    net->template layer_get<0>().init_layer(1, 28, 28, 6, 5, 5);
    net->template layer_get<1>().init_layer(6, 24, 24, 2, 2);
    net->template layer_get<2>().init_layer(6, 12, 12, 16, 5, 5);
    net->template layer_get<3>().init_layer(16, 8, 8, 2, 2);
    net->template layer_get<4>().init_layer(4 * 4 * 16, 150);
    net->template layer_get<5>().init_layer(150, 10);
    net->adam_beta1 = 0.997;
    net->adam_beta2 = 0.997;
    net->learning_rate = 0.1;

    net->display();
    net->display_pretty();
    net->fine_tune(ds.train(), epochs);
    net->evaluate(ds.test());
}


void test_alexnet(size_t epochs) {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    using dbn_t = dll::dbn_desc<
           dll::dbn_layers<
           dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
           dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
           dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
           dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
           dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
           dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
           dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
           dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
           dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
           dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
           dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;

    auto net = std::make_unique<dbn_t>();
    net->template layer_get<0>().init_layer(1, 28, 28, 12, 5, 5);
    net->template layer_get<1>().init_layer(12, 24, 24, 2, 2);
    net->template layer_get<2>().init_layer(12, 12, 12, 12, 3, 3);
    net->template layer_get<3>().init_layer(12, 10, 10, 2, 2);
    net->template layer_get<4>().init_layer(12, 5, 5, 12, 1, 1);
    net->template layer_get<5>().init_layer(12, 5, 5, 12, 1, 1);
    net->template layer_get<6>().init_layer(12, 5, 5, 12, 2, 2);
    net->template layer_get<7>().init_layer(12, 4, 4, 2, 2);
    net->template layer_get<8>().init_layer(12 * 2 * 2, 32);
    net->template layer_get<9>().init_layer(32, 10);

    net->adam_beta1 = 0.997;
    net->adam_beta2 = 0.997;
    net->learning_rate = 0.1;

    net->display();
    net->display_pretty();
    net->fine_tune(ds.train(), epochs);
    net->evaluate(ds.test());
}


void test_vggnet16(size_t epochs) {
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::RELU>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,

        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,

        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;

    auto net = std::make_unique<dbn_t>();
    net->template layer_get<0>().init_layer(1, 28, 28, 12, 5, 5);
    net->template layer_get<1>().init_layer(12, 24, 24, 12, 1, 1);
    net->template layer_get<2>().init_layer(12, 24, 24, 2, 2);
    net->template layer_get<3>().init_layer(12, 12, 12, 24, 3, 3);
    net->template layer_get<4>().init_layer(24, 10, 10, 24, 1, 1);
    net->template layer_get<5>().init_layer(24, 10, 10, 2, 2);
    net->template layer_get<6>().init_layer(24, 5, 5, 24, 1, 1);
    net->template layer_get<7>().init_layer(24, 5, 5, 24, 1, 1);
    net->template layer_get<8>().init_layer(24, 5, 5, 24, 2, 2);
    net->template layer_get<9>().init_layer(24, 4, 4, 2, 2);
    net->template layer_get<10>().init_layer(24, 4, 4, 24, 1, 1);
    net->template layer_get<11>().init_layer(24, 4, 4, 24, 1, 1);
    net->template layer_get<12>().init_layer(24, 4, 4, 24, 1, 1);
    net->template layer_get<13>().init_layer(24, 4, 4, 2, 2);
    net->template layer_get<14>().init_layer(24, 2, 2, 24, 1, 1);
    net->template layer_get<15>().init_layer(24, 2, 2, 24, 1, 1);
    net->template layer_get<16>().init_layer(24, 2, 2, 24, 1, 1);
    net->template layer_get<17>().init_layer(24, 2, 2, 2, 2);
    net->template layer_get<18>().init_layer(24 * 1 * 1, 16);
    net->template layer_get<19>().init_layer(16, 12);
    net->template layer_get<20>().init_layer(12, 10);
    net->adam_beta1 = 0.997;
    net->adam_beta2 = 0.997;
    net->learning_rate = 0.1;

    net->display();
    net->display_pretty();
    net->fine_tune(ds.train(), epochs);
    net->evaluate(ds.test());
}

int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::string test = argv[i];

        if(test.compare("dd") == 0) test_dd(5);
        if(test.compare("ddd") == 0) test_ddd(5);
        if(test.compare("lenet") == 0) test_lenet(5);
        if(test.compare("alexnet") == 0) test_alexnet(5);
        if(test.compare("vggnet16") == 0) test_vggnet16(5);
    }

    return 0;
}