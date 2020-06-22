#ifndef KERAS4DLL_II_NETWORK_TYPES_H
#define KERAS4DLL_II_NETWORK_TYPES_H

#include "dll/dbn.hpp"
#include "dll/network.hpp"
#include "dll/neural/dyn_dense_layer.hpp"


using dense_relu = dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t;
using dense_sig = dll::dyn_dense_layer_desc<dll::activation < dll::function::SIGMOID>>::layer_t;
using dense_soft = dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t;
using dense_id = dll::dyn_dense_layer_desc<dll::activation < dll::function::IDENTITY>>::layer_t;
using dense_tanh = dll::dyn_dense_layer_desc<dll::activation < dll::function::TANH>>::layer_t;


using dbn_dense_RSo =
dll::dbn_desc<
        dll::dbn_layers < dense_relu, dense_soft>,
dll::updater <dll::updater_type::MOMENTUM>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;


#endif //KERAS4DLL_II_NETWORK_TYPES_H
