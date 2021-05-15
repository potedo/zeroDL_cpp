#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "../include/layer.h"

using namespace Eigen;

int main()
{
    using std::cout;
    using std::endl;
    using std::vector;
    using namespace MyDL;

    int n = 1;
    int c = 1;
    int h = 2;
    int w = 2;
    int Fn = 2;
    int Fh = 2;
    int Fw = 2;
    int stride = 1;
    int pad = 1;

    Conv2D conv(c, h, w, Fn, Fh, Fw, stride, pad, 0.1);

    vector<MatrixXd> inputs, outs;
    MatrixXd X = MatrixXd::Zero(n, c*h*w);
    X << 1,2,3,4;
    // X << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;

    inputs.push_back(X);
    outs = conv.forward(inputs);

    cout << outs[0] << endl;

    // ----------------------
    // col2im
    // ----------------------

    int Oh = (2*pad+h - Fh) / stride + 1;
    int Ow = (2*pad+w - Fw) / stride + 1;

    MatrixXd col = MatrixXd::Ones(n*Oh*Ow,c*Fh*Fw);
    MatrixXd img;

    conv.col2im(col, img);

    cout << img << endl;

    // MatrixXd pad_img, img;

    // conv.col2im(col, pad_img);

    // cout << pad_img << endl;

    // conv.suppress(pad_img, img);

    // cout << img << endl;

    vector<MatrixXd> grads, douts;

    douts.push_back(MatrixXd::Ones(n, Fn*Oh*Ow));

    grads = conv.backward(douts);

    cout << grads[0] << endl;

    return 0;
}