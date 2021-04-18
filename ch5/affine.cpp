#include <iostream>
#include <Eigen/Dense>
#include "../simple_lib/include/simple_layer.h"

using namespace Eigen;

int main()
{
    using std::cout;
    using std::endl;

    int batch_size = 3;
    int input_size = 3;
    int output_size = 2;

    MatrixXd X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd Y, dX;
    MatrixXd W = MatrixXd::Zero(input_size, output_size);
    VectorXd b = VectorXd::Zero(output_size);
    MatrixXd dout = MatrixXd::Ones(batch_size, output_size);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    W << 1, 2,
         2, 3,
         3, 4;
    b << 1, 2;

    MyDL::Affine affine(W, b);
    Y = affine.forward(X);
    cout << "forward: " << Y << endl;
    dX = affine.backward(dout);
    cout << "backward: " << dX << endl;

    return 0;
}