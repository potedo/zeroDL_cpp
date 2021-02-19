#include <Eigen/Dense> // Eigenのインクルードパスは、「-I オプションで渡す設定にする」
#include <iostream>

using namespace Eigen;

int main()
{
    using std::cout;
    using std::endl;

    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1.0) + m(0, 1);

    cout << m << endl;

    double *m_array = m.data(); // Eigenの行列を配列化

    cout << "m_array menber[3]: " << m_array[3] << endl;

    cout << sizeof(m_array) / sizeof(m_array[0]) << endl;

    cout << m.array() + 1 << endl;

    // 要素ごとに比較演算子を適用
    Matrix<bool, 2, 2> B = m.unaryExpr([](double p) {return (p > 0.0) ? true : false; });

    cout << B << endl;

    cout << m.unaryExpr([](double p) { return (p > 0.0) ? true : false; }) << endl;

    return 0;
}