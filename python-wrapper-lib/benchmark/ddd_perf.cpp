#include "common.h"

using dbn_3dense = dll::dbn_desc<
        dll::dbn_layers <
        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
        dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
        dll::updater <dll::updater_type::MOMENTUM>,
        dll::trainer <dll::sgd_trainer>,
        dll::shuffle,
        dll::batch_size<100>>
        ::dbn_t;

std::unique_ptr<dbn_3dense> getNet() {
   auto net = std::make_unique<dbn_3dense>();
   net->template layer_get<0>().init_layer(28 * 28, 32);
   net->template layer_get<1>().init_layer(32, 16);
   net->template layer_get<2>().init_layer(16, 10);
   net->initial_momentum = 0.85;
   return net;
}

void perfInit(size_t loops, std::ofstream & file) {
    std::vector<float> durations;
    for (size_t i = 0; i < loops; i++) {
        time_point start = myclock::now();

        auto net = std::make_unique<dbn_3dense>();
        net->template layer_get<0>().init_layer(28 * 28, 32);
        net->template layer_get<1>().init_layer(32, 16);
        net->template layer_get<2>().init_layer(16, 10);

        net->initial_momentum = 0.85;

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("3xdense_net", "perf_init", file, durations);
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
    print("3xdense_net", "perf_display", file, durations);
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
    print("3xdense_net", "perf_display_pretty", file, durations);
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
    print("3xdense_net", "perf_train", file, durations);
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
    print("3xdense_net", "perf_evaluate", file, durations);
}

void perfAll(size_t loops, std::ofstream & file, size_t epochs) {
    std::vector<float> durations;
    auto ds = dll::make_mnist_dataset(dll::batch_size < 100 > {}, dll::scale_pre < 255 > {});

    for (size_t i = 0; i < loops; i++) {
        time_point start = myclock::now();
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
        net->fine_tune(ds.train(), epochs);
        net->evaluate(ds.test());

        time_point end = myclock::now();
        durations.push_back(std::chrono::duration_cast<resolution>(end - start).count());
    }
    print("3xdense_net", "perf_all", file, durations);
}

int main(int, char**) {
    std::ofstream file("../benchmark/benchmark_cpp_ddd.txt",  std::ofstream::out | std::ofstream::trunc);
    file.clear();

    perfInit(10000, file);
    perfDisplay(10000, file);
    perfDisplayPretty(10000, file);
    perfTrain(50, file, 25);
    perfEvaluate(50, file, 25);
    perfAll(50,file,25);
    /*perfInit(100, file);
    perfDisplay(100, file);
    perfDisplayPretty(100, file);
    perfTrain(2, file, 2);
    perfEvaluate(2, file, 2);
    perfAll(2,file,2);*/

    file.close();
    return 0;
}
