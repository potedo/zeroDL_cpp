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
        static double max_coeff = x.maxCoeff();
        x = x.unaryExpr([] (double p){return exp(p - max_coeff); });
        x.array() /= x.sum();
        return x;
    }    

}

#endif // _SIMPLE_ACTIVATION_H_