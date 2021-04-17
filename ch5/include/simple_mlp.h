// #ifndef _SIMPLE_MLP_H_
// #define _SIMPLE_MLP_H_

// #include <Eigen/Dense>
// #include <string>
// #include <map>
// #include "simple_layer.h"

// namespace MyDL{
    
//     using namespace Eigen;
//     using std::string;
//     using std::map;

//     class TwoLayerNetwork:{
//         private:
//             int _input_size;
//             int _hidden_size;
//             int _output_size;
//             map<string, MatrixXd> _params;
//             map<string, MatrixXd> _cache;
//             map<string, > _layers;

//         public:
//             TwoLayerNetwork(int input_size=1, int hidden_size=1, int output_size=1, double weight_init_std=0.01);
//             MatrixXd predict(MatrixXd&);
//             double loss(MatrixXd&, MatrixXd&);
//             double accuracy(MatrixXd&, MatrixXd&);
//             map<string, MatrixXd> gradient(MatrixXd&, MatrixXd&);

//     };
// }


// #endif // _SIMPLE_MLP_H_