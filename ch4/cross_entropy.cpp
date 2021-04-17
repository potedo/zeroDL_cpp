#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

double categorical_cross_entropy(VectorXd, VectorXd);

int main(){
    using std::cout;
    using std::endl;

    VectorXd t = VectorXd::Zero(10);
    VectorXd y = VectorXd::Zero(10);
    double cross_entropy;

    t(2) = 1;
    y << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;

    cross_entropy = categorical_cross_entropy(t, y);
    cout << "cross entropy: " << cross_entropy << endl;

    return 0;
}

double categorical_cross_entropy(VectorXd t, VectorXd y){
    double delta = 1.0e-7;
    return -(t.array() * (y.array() + delta).log()).sum();
}