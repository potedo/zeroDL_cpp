#include <iostream>
#include <Eigen/Dense>
#include "include/simple_layer.h"

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;
    using namespace MyDL;

    int batch_size = 2;
    int num_category = 3;
    MatrixXd X = MatrixXd::Zero(batch_size, num_category);
    MatrixXd t = MatrixXd::Zero(batch_size, num_category);
    double loss;
    MatrixXd dX;

    // X << 0.2, 0.3, 0.5,
    //      0.1, 0.7, 0.2;
    X << 1, 2, 3,
         1, 0, 0;
    t << 0, 0, 1,
         1, 0, 0;

    SoftmaxWithLoss loss_layer;

    loss = loss_layer.forward(X, t);
    cout << "loss: " << loss << endl; 

    dX = loss_layer.backward();
    cout << "gradient: " << endl;
    cout << dX << endl;

    return 0;
}