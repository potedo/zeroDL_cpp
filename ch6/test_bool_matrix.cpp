#include <iostream>
#include <Eigen/Dense>

int main()
{
    using std::cout;
    using std::endl;
    using namespace Eigen;

    MatrixXd A = MatrixXd::Random(3,3);

    Matrix<bool, Dynamic, Dynamic> B = A.array() < 0.5;

    MatrixXd C;

    C = B.cast<double>();

    cout << "Matrix A: " << endl;
    cout << A << endl;

    cout << "Bool Matrix B:" << endl;
    cout << B << endl;

    cout << "double Matrix C:" << endl;
    cout << C << endl;

    cout << "double Matrix A mul C" << endl;
    cout << A.array() * C.array() << endl;

    return 0;
}