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

    MatrixXd softmax(MatrixXd); // 関数の定義をヘッダ内に書くとコンパイルエラーになる
    MatrixXd sigmoid(MatrixXd); // テンプレートは実装を書いていても問題ない？

}

#endif // _SIMPLE_ACTIVATION_H_