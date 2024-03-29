#include <iostream>
#include <Eigen/Dense>
#include "../simple_lib/include/simple_layer.h"

using namespace Eigen;
using namespace MyDL;

int main()
{
    using std::cout;
    using std::endl;

    MatrixXd x = MatrixXd::Zero(2, 2);
    MatrixXd y, dx;
    MatrixXd dout = MatrixXd::Ones(2, 2);
    x <<   1, -0.5,
        -2.0, 3.0;

    Sigmoid sigmoid;
    y = sigmoid.forward(x);
    cout << "forward: " << y << endl;
    dx = sigmoid.backward(dout);
    cout << "backward: " << dx << endl;

    return 0;
}