#ifndef _SIMPLE_LAYER_H_
#define _SIMPLE_LAYER_H_

#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;

    // シンプルな乗算 → 行列同士の積の場合はアダマール積として対処
    class MulLayer{
        private:
            MatrixXd _x; // 逆伝播で使用するために内部で保持
            MatrixXd _y; // 同上 → 推論モードの時は保持しない仕組みを用意したい
        public:
            MatrixXd forward(MatrixXd&, MatrixXd&);
            void backward(MatrixXd&, MatrixXd&, MatrixXd&);
    };


    class AddLayer{
        // この層に関しては、微分をそのまま流すので内部に順伝播の情報を保持する必要はない
        private:

        public:
            MatrixXd forward(MatrixXd&, MatrixXd&);
            void backward(MatrixXd&, MatrixXd&, MatrixXd&);
    };


    class ReLU{
        private:
            MatrixXd mask;
        public:
            MatrixXd forward(MatrixXd&);
            MatrixXd backward(MatrixXd&);
    };


    class Sigmoid{
        private:
            MatrixXd _y;
        public:
            MatrixXd forward(MatrixXd&);
            MatrixXd backward(MatrixXd&);
    };


    class Affine{
        private:
            MatrixXd _X;
            MatrixXd _W, dW;
            VectorXd _b, db;
        public:
            Affine();
            Affine(MatrixXd&, VectorXd&);
            MatrixXd forward(MatrixXd&);
            MatrixXd backward(MatrixXd&);
    };

    
    class SoftmaxWithLoss{
        private:
            double _loss;
            MatrixXd _Y;
            MatrixXd _t;
        public:
            double forward(MatrixXd&, MatrixXd&);
            MatrixXd backward(double dout=1);
    };

}

#endif // _SIMPLE_LAYER_H_