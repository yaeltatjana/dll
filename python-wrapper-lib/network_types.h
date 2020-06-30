#ifndef KERAS4DLL_II_NETWORK_TYPES_H
#define KERAS4DLL_II_NETWORK_TYPES_H

#include "dll/dbn.hpp"
#include "dll/network.hpp"
#include "dll/neural/dyn_dense_layer.hpp"
#include "dll/neural/dyn_conv_layer.hpp"
#include "dll/pooling/dyn_mp_layer.hpp"
#include "dll/dbn.hpp"



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


using dbn_3dense_RRSo =
dll::dbn_desc<
        dll::dbn_layers < dense_relu,dense_relu, dense_soft>,
dll::updater <dll::updater_type::MOMENTUM>,
dll::trainer <dll::sgd_trainer>,
dll::shuffle,
dll::batch_size<100>>
::dbn_t;

using dbn_lenet = dll::dbn_desc<
        dll::dbn_layers<
        dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
dll::dyn_mp_2d_layer_desc<dll::weight_type<float>>::layer_t,
//dll::dyn_conv_layer_desc<dll::activation<dll::function::TANH>>::layer_t,
dll::dyn_dense_layer_desc<dll::activation < dll::function::RELU>>::layer_t,
dll::dyn_dense_layer_desc<dll::activation < dll::function::SOFTMAX>>::layer_t>,
dll::trainer<dll::sgd_trainer>, dll::updater<dll::updater_type::NADAM>, dll::batch_size<100>>::dbn_t;

using dbn_alexnet = dll::dbn_desc<
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


using dbn_vggnet = dll::dbn_desc<
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




#endif //KERAS4DLL_II_NETWORK_TYPES_H
