#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/neural/dyn_conv_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "dll/datasets.hpp"
#include "dll/network.hpp"


using myclock = std::chrono::system_clock;              ///<    clock of the system
using time_point = std::chrono::time_point<myclock>;    ///<    time point of the chrono
using resolution = std::chrono::milliseconds;           ///<    precision

/**
 * Function to print the results of a benchmark function
 * @param net           name of network
 * @param func_name     name of tested function
 * @param file          file where to print the output
 * @param durations     table of durations
 */
void print(std::string net, std::string func_name, std::ofstream &file, std::vector<float> &durations) {
    double avg = std::accumulate(durations.begin(), durations.end(), 0) / durations.size();
    double min = *std::min_element(durations.begin(), durations.end());
    double max = *std::max_element(durations.begin(), durations.end());
    file << "======================= [ "
         << net << " - "
         << func_name << " ] =======================\nAverage: "
         << avg << " ms\nMax: "
         << max << " ms\nMin: "
         << min << " ms\n"
         << std::endl;
}