#include <iostream>
#include <Eigen/Dense>
#include "../ch5/include/simple_activation.h"

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;

    MatrixXd a = MatrixXd::Zero(3, 1);
    MatrixXd s1 = MatrixXd::Zero(3, 1);
    MatrixXd s2 = MatrixXd::Zero(3, 1);
    a << 1010, 1000, 990;

    s1 = a.array().exp();
    s1 /= s1.sum();

    cout << s1 << endl;

    s2 = MyDL::softmax(a);

    cout << s2 << endl;

    return 0;
}