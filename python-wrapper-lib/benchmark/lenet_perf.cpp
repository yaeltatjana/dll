#include "common.h"


using dbn_lenet = dll::dbn_desc<
        dll::dbn_layers<
        dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
        dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
        dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::TANH>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
        dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;


std::unique_ptr<dbn_lenet> getNet() {
    auto net = std::make_unique<dbn_lenet>();
    net->template layer_get<0>().init_layer(1, 28, 28, 6, 5, 5);
    net->template layer_get<1>().init_layer(6, 24, 24, 2, 2);
    net->template layer_get<2>().init_layer(6, 12, 12, 16, 5, 5);
    net->template layer_get<3>().init_layer(16, 8, 8, 2, 2);
    net->template layer_get<4>().init_layer(4 * 4 * 16, 150);
    net->template layer_get<5>().init_layer(150, 10);
    net->adam_beta1 = 0.997;
    net->adam_beta2 = 0.997;
    net->learning_rate = 0.1;
   return net;
}


void perfInit(size_t loops, std::ofstream & file) {
    std::vector<float> durations;
    for (size_t i = 0; i < loops; i++) {
        time_point start = myclock::now();

        auto net = std::make_unique<dbn_lenet>();
        net->template layer_get<0>().init_layer(1, 28, 28, 6, 5, 5);
        net->template layer_get<1>().init_layer(6, 24, 24, 2, 2);
        net->template layer_get<2>().init_layer(6, 12, 12, 16, 5, 5);
        net->template layer_get<3>().init_layer(16, 8, 8, 2, 2);
        net->template layer_get<4>().init_layer(4 * 4 * 16, 150);
        net->template layer_get<5>().init_layer(150, 10);
        net->adam_beta1 = 0.997;
        net->adam_beta2 = 0.997;
        net->learning_rate = 0.1;

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_init", file, durations);
}

void perfDisplay(size_t loops, std::ofstream & file) {
    std::vector<float> durations;
    for (size_t i = 0; i < loops; i++) {
        auto net = getNet();

        time_point start = myclock::now();
        net->display();

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_display", file, durations);
}

void perfDisplayPretty(size_t loops, std::ofstream & file) {
    std::vector<float> durations;
    for (size_t i = 0; i < loops; i++) {
        auto net = getNet();

        time_point start = myclock::now();
        net->display_pretty();

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_display_pretty", file, durations);
}

void perfTrain(size_t loops, std::ofstream & file, size_t epochs) {
    std::vector<float> durations;
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    for (size_t i = 0; i < loops; i++) {
        auto net = getNet();

        time_point start = myclock::now();
        net->fine_tune(ds.train(), epochs);

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_train", file, durations);
}

void perfEvaluate(size_t loops, std::ofstream & file, size_t epochs) {
    std::vector<float> durations;
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    for (size_t i = 0; i < loops; i++) {
        auto net = getNet();

        net->fine_tune(ds.train(), epochs);

        time_point start = myclock::now();
        net->evaluate(ds.test());
        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_evaluate", file, durations);
}

void perfAll(size_t loops, std::ofstream & file, size_t epochs) {
    std::vector<float> durations;
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    for (size_t i = 0; i < loops; i++) {
        time_point start = myclock::now();
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
        net->fine_tune(ds.train(), epochs);
        net->evaluate(ds.test());

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("dbn_lenet", "perf_all", file, durations);
}

int main(int, char**) {
    std::ofstream file("../benchmark/benchmark_cpp_lenet.txt",  std::ofstream::out | std::ofstream::trunc);
    file.clear();

   /* perfInit(10000, file);
    perfDisplay(10000, file);
    perfDisplayPretty(10000, file);
    perfTrain(50, file, 25);*/
    perfEvaluate(50, file, 25);
    /*perfAll(50,file,25);
    perfInit(100, file);
    perfDisplay(100, file);
    perfDisplayPretty(100, file);
    perfTrain(2, file, 5);
    perfEvaluate(2, file, 5);
    perfAll(2,file,5);*/

    file.close();
    return 0;
}