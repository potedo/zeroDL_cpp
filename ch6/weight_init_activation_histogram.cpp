#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <Eigen/Dense>
#include "../matplotlibcpp.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

MatrixXd sigmoid(MatrixXd&);
// MatrixXd ReLU(MatrixXd&);
// MatrixXd tanh(MatrixXd&);

int main()
{
    using std::cout;
    using std::endl;
    using std::map;
    using std::vector;
    using std::string;

    int node_num = 100;
    MatrixXd input_data = MatrixXd::Random(1000, node_num);
    int hidden_layer_size = 5;
    map<int, MatrixXd> activations;

    MatrixXd x = input_data;
    MatrixXd w, a, z;

    for (int i = 0; i < hidden_layer_size; i++)
    {
        if (i != 0)
        {
            x = activations[i-1];
        }

        w = MatrixXd::Random(node_num, node_num) * 0.1; // ここの重みを変更する
        a = x * w;
        z = sigmoid(a);

        activations[i] = z;
    }

    // vectorにEigenを詰めなおして、プロットを作成すること
    // for (auto activation : activations)
    // {
    //     cout << activation.first << ":" << endl;
    //     cout << activation.second << endl;
    // }

    vector<double> a_vec(1000*node_num);
    Map<MatrixXd>(&a_vec[0], 1000, node_num) = activations[0];

    int bins = 30;
    string title = "Layer1";

    plt::hist(a_vec, bins);
    plt::xlim(0, 1);
    plt::title(title);
    plt::save("activation_hist_xavier0.png");

    // vector<double> tmp_a(node_num*node_num);
    for (auto activation : activations)
    {
        if (activation.first != 0)
        {
            string save_filename = "activation_hist_xavier";
            save_filename += std::to_string(activation.first);
            save_filename += ".png";

            Map<MatrixXd>(&a_vec[0], 1000, node_num) = activation.second;

            title = "Layer";
            title += std::to_string(activation.first + 1);
            plt::hist(a_vec, bins);
            plt::xlim(0, 1);
            plt::title(title);
            plt::save(save_filename);
        }
    }

    return 0;
}

MatrixXd sigmoid(MatrixXd& X)
{
    return X.unaryExpr([](double p){return 1 / (1 + exp(p));});
}

// MatrixXd ReLU(MatrixXd& X)
// {
//     return X.unaryExpr([](double p){return std::max(0, p);});
// }

// MatrixXd tanh(MatrixXd& X)
// {
//     return X.unaryExpr([](double p){return tanh(p);});
// }