#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

double mean_squared_error(VectorXd, VectorXd);

int main(){
    using std::cout;
    using std::endl;

    VectorXd t = VectorXd::Zero(10);
    VectorXd y = VectorXd::Zero(10);
    double mse;

    t(2) = 1;
    y << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;

    mse = mean_squared_error(t, y);
    cout << "mean squared error: " << mse << endl;

    return 0;
}

double mean_squared_error(VectorXd t, VectorXd y){
    return (t - y).array().square().sum() / 2;
}