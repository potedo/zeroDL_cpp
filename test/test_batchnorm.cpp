#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "../include/layer.h"

using namespace Eigen;

int main()
{
    using std::cout;
    using std::endl;
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using std::make_shared;
    using namespace MyDL;

    int dim = 2;
    int batch_size = 3;

    // auto pgamma = make_shared<MatrixXd>(1, dim);
    // auto pbeta  = make_shared<MatrixXd>(1, dim);

    // *pgamma = MatrixXd::Random(1, dim);
    // *pbeta  = MatrixXd::Random(1, dim);

    // MyDL::BatchNorm batch_norm(pgamma, pbeta);

    BatchNorm batch_norm(dim);

    vector<MatrixXd> inputs, outputs, douts;
    MatrixXd X = MatrixXd::Random(batch_size, dim);
    inputs.push_back(X);

    MatrixXd dout = MatrixXd::Random(batch_size, dim);
    douts.push_back(dout);

    bool train_flg = Config::getInstance().get_flag();

    cout << "Train Flag: " << train_flg << endl;
    cout << "--- Mode Train ---" << endl;

    outputs = batch_norm.forward(inputs);

    cout << "Forward Result: " << outputs[0] << endl;

    outputs = batch_norm.forward(inputs); // _avg_mean, _avg_varをもう一段階変化させる

    Config::getInstance().set_flag(false);
    train_flg = Config::getInstance().get_flag();

    cout << "Train Flag: " << train_flg << endl;
    cout << "--- Mode Inference ---" << endl;

    outputs = batch_norm.forward(inputs);

    cout << "Forward Result: " << endl;
    cout << outputs[0] << endl;


    Config::getInstance().set_flag(true);
    train_flg = Config::getInstance().get_flag();

    cout << "Train Flag: " << train_flg << endl;
    cout << "--- Mode Train ---" << endl;

    vector<MatrixXd> grads;

    grads = batch_norm.backward(douts);
    cout << "Backward Result: " << endl;;
    cout << grads[0] << endl;

    return 0;
}