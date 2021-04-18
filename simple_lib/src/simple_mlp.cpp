// #include <map>
// #include <string>
// #include <Eigen/Dense>
// #include "simple_layer.h"
// #include "simple_mlp.h"

// namespace MyDL{

//     using namespace Eigen;

//     TwoLayerNetwork::TwoLayerNetwork(int input_size, int hidden_size, int output_size, double weight_init_std){
//         _input_size = input_size;
//         _hidden_size = hidden_size;
//         _output_size = output_size;

//         _params["W1"] = weight_init_std * MatrixXd::Random(input_size, hidden_size);
//         _params["b1"] = VectorXd::Zero(hidden_size);
//         _params["W2"] = weight_init_std * MatrixXd::Random(hidden_size, output_size);
//         _params["b1"] = VectorXd::Zero(output_size);

        
//     }

//     MatrixXd TwoLayerNetwork::predict(MatrixXd& X){

//     }



// }