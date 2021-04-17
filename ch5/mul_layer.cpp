#include <iostream>
#include <Eigen/Dense>
#include "include/simple_layer.h"

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;
    using namespace MyDL;

    MatrixXd x = MatrixXd::Zero(2,2);
    MatrixXd y = MatrixXd::Zero(2,2);
    MatrixXd z, dx, dy;
    MatrixXd dout = MatrixXd::Ones(2, 2);

    x << 1, 2,
         3, 4;
    y << 2, 2,
         3, 3;

    MulLayer mul;

    z = mul.forward(x, y);
    mul.backward(dout, dx, dy);

    cout << "--- forward ---" << endl;
    cout << z << endl;

    cout << "--- backward ---" << endl;
    cout << dx << endl;
    cout << dy << endl;

    return 0;
}