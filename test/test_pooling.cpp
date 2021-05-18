#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "../include/layer.h"

int main()
{
    using namespace Eigen;
    using std::cout;
    using std::endl;
    using namespace MyDL;

    int n = 1;
    int c = 1;
    int h = 3;
    int w = 3;
    int Fn = 2;
    int Fh = 2;
    int Fw = 2;
    int stride = 1;
    int pad = 1;

    Conv2D conv(c, h, w, Fn, Fh, Fw, stride, pad, 0.1);

    vector<MatrixXd> inputs, conv_outs, pooling_outs;
    MatrixXd X = MatrixXd::Zero(n, c * h * w);
    X << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    n = 1;
    c = Fn;
    h = (pad*2+h-Fh)/stride + 1;
    w = (pad*2+w-Fw)/stride + 1;
    int Ph = 2;
    int Pw = 2;
    stride = 2;
    pad = 0;

    int Oh = (2*pad + h - Ph) / stride + 1;
    int Ow = (2*pad + w - Pw) / stride + 1;

    Pooling pooling(c, h, w, Ph, Pw, stride, pad);

    // MatrixXd X = MatrixXd::Zero(n, h*w*c);

    // X << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
    // X << 1,2,3,4,5,1,7,1,9,10,11,12,13,1,15,1;

    // vector<MatrixXd> inputs, outputs;
    // inputs.push_back(X);

    // outputs = pooling.forward(inputs);

    // cout << outputs[0] << endl;

    // vector<MatrixXd> douts, grads;
    // MatrixXd dout = MatrixXd::Zero(n, c*Oh*Ow);

    // dout << 1,2,3,4;

    // douts.push_back(dout);

    // grads = pooling.backward(douts);

    // cout << grads[0] << endl;

    // -----------------------
    //   Conv2Dとの結合の動作確認
    // -----------------------

    inputs.push_back(X);
    conv_outs = conv.forward(inputs);

    cout << conv_outs[0] << endl;

    pooling_outs = pooling.forward(conv_outs);

    cout << pooling_outs[0] << endl;

    cout << pooling._argmax << endl;

    vector<MatrixXd> douts, pooling_grads, conv_grads;
    MatrixXd dout = MatrixXd::Ones(n, Oh*Ow*c);
    dout << 1,2,3,4,1,2,3,4;
    douts.push_back(dout);

    pooling_grads = pooling.backward(douts);

    cout << pooling_grads[0] << endl;

    conv_grads = conv.backward(pooling_grads);

    cout << conv_grads[0] << endl;

    return 0;
}