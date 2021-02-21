#ifndef _SIMPLE_ACTIVATION_H_
#define _SIMPLE_ACTIVATION_H_

#include <algorithm>
#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;


    template<typename T>
    int step_function(T x){
        bool y = x > 0;
        return int(y);
    }

    template<typename T>
    double sigmoid(T x){
        double ret = 1 / (1 + exp(-x));
        return ret;
    }

    template<typename T>
    double relu(T x){
        double ret = std::max<double>(0, x); // std::maxはテンプレートライブラリなので型指定必要
        return ret;
    }

    template<typename T>
    T identity_function(T x){
        return x;
    }

    // 実装がめんどくさいので直接Eigen::MatrixXdを引数にとるよう実装
    MatrixXd softmax(MatrixXd x){
        VectorXd max_coeff_vec, rowwise_sum;
        max_coeff_vec = x.rowwise().maxCoeff();

        x = x.colwise() - max_coeff_vec; // expのオーバーフロー回避 → 各バッチベクトルごとに最大の要素を抽出
        x = x.array().exp(); // expを各要素に実行
        rowwise_sum = x.rowwise().sum(); // バッチベクトルごとに総和を計算

        x.array().colwise() /= rowwise_sum.array(); // 出力の総和が1になるよう調整
        return x;
    }

    MatrixXd sigmoid(MatrixXd x){
        x = x.unaryExpr([] (double p){return 1 / (1 + exp(-p));});
        return x;
    }

}

#endif // _SIMPLE_ACTIVATION_H_