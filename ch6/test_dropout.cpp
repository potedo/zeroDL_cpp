#include <iostream>
#include <Eigen/Dense>
#include "../include/layer.h"

int main()
{
    using std::cout;
    using std::endl;
    using namespace MyDL;
    using namespace Eigen;

    int row = 4;
    int col = 5;
    double dropout_ratio = 0.5;
    Dropout dropout(row, col, dropout_ratio);

    MatrixXd X = MatrixXd::Random(row, col);
    MatrixXd dout = MatrixXd::Ones(row, col);
    vector<MatrixXd> inputs, outputs, douts, grads;
    inputs.push_back(X);
    douts.push_back(dout);

    cout << "input: " << endl;
    cout << X << endl;

    cout << "Dropout forward -train mode- :" << endl;
    outputs = dropout.forward(inputs);

    cout << outputs[0] << endl;

    Config::getInstance().set_flag(false);

    cout << "Dropout forward -inference mode- : " << endl;
    outputs = dropout.forward(inputs);

    cout << outputs[0] << endl;

    cout << "Dropout backward: " << endl;
    grads = dropout.backward(douts);

    cout << grads[0] << endl;

    return 0;
}