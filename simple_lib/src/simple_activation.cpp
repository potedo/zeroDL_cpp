#include <Eigen/Dense>
#include "simple_activation.h"

namespace MyDL{
    using namespace Eigen;

    // 実装がめんどくさいので直接Eigen::MatrixXdを引数にとるよう実装
    MatrixXd softmax(MatrixXd x)
    {
        VectorXd max_coeff_vec, rowwise_sum;
        max_coeff_vec = x.rowwise().maxCoeff();

        x = x.colwise() - max_coeff_vec; // expのオーバーフロー回避 → 各バッチベクトルごとに最大の要素を抽出
        x = x.array().exp();             // expを各要素に実行
        rowwise_sum = x.rowwise().sum(); // バッチベクトルごとに総和を計算

        x.array().colwise() /= rowwise_sum.array(); // 出力の総和が1になるよう調整
        return x;
    }

    MatrixXd sigmoid(MatrixXd x)
    {
        x = x.unaryExpr([](double p) { return 1 / (1 + exp(-p)); });
        return x;
    }
}