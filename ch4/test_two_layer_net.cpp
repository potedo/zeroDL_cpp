#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "../ch5/include/two_layer_net.h"

using namespace Eigen;
using namespace MyDL;

int main(){
    using std::cout;
    using std::endl;
    using std::string;
    using std::map;

    int batch_size = 2;
    int input_size = 2;
    int hidden_size = 3;
    int output_size = 2;

    // int batch_size = 3;
    // int input_size = 2;
    // int hidden_size = 3;
    // int output_size = 5;

//     int batch_size = 5;
//     int input_size = 28 * 28;
//     int hidden_size = 100;
//     int output_size = 10;

    TwoLayerNet net(input_size, hidden_size, output_size, 0.01);

    MatrixXd X = MatrixXd::Random(batch_size, input_size);
    MatrixXd y; // (batch_size, output_size)
    MatrixXd t = MatrixXd::Zero(batch_size, output_size);
//     MatrixXd t = MatrixXd::Random(batch_size, output_size);
    double loss;
    double accuracy;
    map<string, MatrixXd> grads, numerical_grads;

    X = -2 * MatrixXd::Identity(2, 2) + MatrixXd::Ones(2, 2);
    t = MatrixXd::Identity(2, 2);

    // X << 1, 2,
    //      3, 4,
    //      5, 6;
    // t << 0, 1, 0, 0, 0,
    //      0, 0, 0, 1, 0,
    //      0, 0, 0, 1, 0;
    
    y = net.predict(X);
    cout << "y = " << y << endl; // predictの結果は、行方向に総和を取った場合、「1」となっていることを確認

    loss = net.loss(X, t);
    cout << "loss: " << loss << endl;

    accuracy = net.accuracy(X, t);
    cout << "accuracy: " << accuracy << endl; // この入力の場合、accuracyは0.666..が正解

    grads = net.gradient(X, t);
    cout << "gradient of W1: " << grads["W1"] << endl;
    cout << "gradient of W2: " << grads["W2"] << endl;
    cout << "gradient of b1: " << grads["b1"] << endl;
    cout << "gradient of b2: " << grads["b2"] << endl;

    cout << "numerical gradient" << endl;
    numerical_grads = net.numerical_gradient(X, t);
    cout << "gradient of W1: " << numerical_grads["W1"] << endl;
    cout << "gradient of W2: " << numerical_grads["W2"] << endl;
    cout << "gradient of b1: " << numerical_grads["b1"] << endl;
    cout << "gradient of b2: " << numerical_grads["b2"] << endl;

    return 0;
}