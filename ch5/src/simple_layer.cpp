#include <Eigen/Dense>
#include "simple_layer.h"
#include "simple_activation.h"
#include "simple_loss.h"

namespace MyDL{

    using namespace Eigen;

    // -------------------------------------------------
    //          MulLayer
    // -------------------------------------------------
    MatrixXd MulLayer::forward(MatrixXd& x, MatrixXd& y){
        _x = x;
        _y = y;

        return x.array() * y.array();
    }

    void MulLayer::backward(MatrixXd& dout, MatrixXd& dx, MatrixXd& dy){
        dx = dout.array() * _y.array();
        dy = dout.array() * _x.array();
    }

    // -------------------------------------------------
    //          AddLayer
    // -------------------------------------------------
    MatrixXd AddLayer::forward(MatrixXd& x, MatrixXd& y){
        return x + y;
    }

    void AddLayer::backward(MatrixXd& dout, MatrixXd& dx, MatrixXd& dy){
        dx = dout;
        dy = dout;        
    }

    // -------------------------------------------------
    //          ReLU
    // -------------------------------------------------
    MatrixXd ReLU::forward(MatrixXd& x){
        mask = x.unaryExpr([](double p){return p >= 0;}).cast<double>();
        return x.array() * mask.array();
    }

    MatrixXd ReLU::backward(MatrixXd& dout){
        return dout.array() * mask.array();
    }

    // -------------------------------------------------
    //          Sigmoid
    // -------------------------------------------------
    MatrixXd Sigmoid::forward(MatrixXd& x){
        _y = x.unaryExpr([](double p){return 1/(1 + exp(-p));});
        return _y; // 内部変数に格納するの忘れがち
    }

    MatrixXd Sigmoid::backward(MatrixXd& dout){
        return dout.array() * (MatrixXd::Ones(dout.rows(), dout.cols()).array() - _y.array()) * _y.array();
    }

    // -------------------------------------------------
    //          Affine
    // -------------------------------------------------
    Affine::Affine(MatrixXd& W, VectorXd& b){
        _W = W;
        _b = b;
    }

    MatrixXd Affine::forward(MatrixXd& X){
        MatrixXd Y;
        _X = X; // 内部変数に格納するの忘れがち
        Y = (X * _W).rowwise() + _b.transpose();
        return Y;
    }

    MatrixXd Affine::backward(MatrixXd& dout){
        MatrixXd dX;
        dX = dout * _W.transpose();
        dW = _X.transpose() * dout;
        db = dout.colwise().sum();
        return dX;
    }

    // -------------------------------------------------
    //          SoftmaxWithLoss
    // -------------------------------------------------
    double SoftmaxWithLoss::forward(MatrixXd& X, MatrixXd& t){
        _t = t;
        _Y = softmax(X);
        _loss = cross_entropy_error(_Y, t);
        return _loss;
    }

    MatrixXd SoftmaxWithLoss::backward(double dout){
        double batch_size = _t.rows();
        MatrixXd dx;
        dx = (_Y - _t) / batch_size;

        return dx;
    }
    

}