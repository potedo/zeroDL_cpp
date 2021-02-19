#ifndef _SIMPLE_PERCEPTRON_H_
#define _SIMPLE_PERCEPTRON_H_

namespace MyDL{

    template <typename T>
    int AND(T x1, T x2){
        double w1 = 0.5;
        double w2 = 0.5;
        // double theta = 0.7; // 閾値
        double b = -0.7;

        double tmp = w1 * x1 + w2 * x2 + b;

        // if (tmp <= theta){
        //     return 0;
        // } else {
        //     return 1;
        // }

        if (tmp <= 0){
            return 0;
        } else {
            return 1;
        }
    }

    template <typename T>
    int NAND(T x1, T x2){
        double w1 = -0.5;
        double w2 = -0.5;
        double b = 0.7;

        double tmp = w1 * x1 + w2 * x2 + b;

        if (tmp <= 0){
            return 0;
        } else {
            return 1;
        }
    }

    template <typename T>
    int OR(T x1, T x2){
        double w1 = 0.5;
        double w2 = 0.5;
        double b = -0.3;

        double tmp = w1 * x1 + w2 * x2 + b;

        if (tmp <= 0){
            return 0;
        } else {
            return 1;
        }
    }

    template <typename T>
    int NOR(T x1, T x2)
    {
        double w1 = -0.5;
        double w2 = -0.5;
        double b = 0.2;

        double tmp = w1 * x1 + w2 * x2 + b;

        if (tmp <= 0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    template <typename T>
    int XOR(T x1, T x2){
        double h1, h2, out;
        h1 = OR<T>(x1, x2);
        h2 = NAND<T>(x1, x2);
        out = AND(h1, h2);
        return out;
    }
}


#endif // _SIMPLE_PERCEPTRON_H_