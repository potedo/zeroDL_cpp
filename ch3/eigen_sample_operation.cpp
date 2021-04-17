#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;

    MatrixXd A = MatrixXd::Zero(2,2);
    MatrixXd B = MatrixXd::Zero(2,2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    cout << "Same Dimension Matrix Multiply" << endl;
    cout << A.rows() << " " << A.cols() << endl;
    cout << A*B << endl; // 行列の積
    cout << A.array() * B.array() << endl; // 要素積

    MatrixXd X = MatrixXd::Zero(2,3);
    MatrixXd Y = MatrixXd::Zero(3,2);

    X << 1,2,3, 4,5,6;
    Y << 1,2, 3,4, 5,6;

    cout << "Different Dimension Matrix Multiply" << endl;
    cout << "X shape: " << X.rows() << " " << X.cols() << endl;
    cout << "Y shape: " << Y.rows() << " " << Y.cols() << endl;
    cout << X * Y << endl;
    cout << X.array() * Y.transpose().array() << endl;

    return 0;
}