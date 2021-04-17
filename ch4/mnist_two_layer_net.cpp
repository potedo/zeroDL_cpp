#include <map>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "../ch5/include/two_layer_net.h"
#include "../datasets/include/mnist.h"

using namespace Eigen;

int main(){
    using std::map;
    using std::cout;
    using std::endl;
    using std::string;
    using namespace MyDL;

    int batch_size = 5;
    int input_size = 28*28;
    int hidden_size = 100;
    int output_size = 10;

    MnistEigenDataset mnist(batch_size);

    MatrixXd train_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd train_y = MatrixXd::Zero(batch_size, output_size);
    bool one_hot_label = true;

    TwoLayerNet net(input_size, hidden_size, output_size, 0.01);
    mnist.next_train(train_X, train_y, one_hot_label);

    MatrixXd pred_y;
    double loss, accuracy;
    map<string, MatrixXd> grads;

    pred_y = net.predict(train_X);
    loss = net.loss(train_X, train_y);
    accuracy = net.accuracy(train_X, train_y);
    grads = net.gradient(train_X, train_y);

    cout << "Prediction: " << pred_y << endl;
    cout << "Loss: " << loss << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Gradient W1: " << grads["dW1"] << endl;

    return 0;
}