#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "../include/sample_network.h"
#include "../datasets/include/mnist.h"

using namespace Eigen;

int main()
{
    using std::cout;
    using std::endl;
    using std::vector;
    using std::string;
    using std::unordered_map;
    using namespace MyDL;

    int num_iters = 1000;
    double learning_rate = 0.1;

    int batch_size = 100;
    int input_size = 28 * 28;
    int hidden_size = 50;
    int output_size = 10;

    MatrixXd train_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd test_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd train_y = MatrixXd::Zero(batch_size, 10);
    MatrixXd test_y = MatrixXd::Zero(batch_size, 10);

    MnistEigenDataset mnist(batch_size);

    TwoLayerNet net(input_size, hidden_size, output_size);

    vector<MatrixXd> inputs, loss, val_inputs;
    unordered_map<string, MatrixXd> grads;
    double accuracy;

    mnist.next_train(train_X, train_y, true);
    inputs.push_back(train_X);
    inputs.push_back(train_y);

    grads = net.gradient(inputs);

    // cout << "--- parameter b1 ---" << endl;
    // cout << *(net.params["b1"]) << endl;
    // cout << "--- parameter b1 update ---" << endl;
    // *(net.params["b1"]) -= -learning_rate * grads["b1"];
    // MatrixXd dParam = grads["b1"];
    // *(net.params["b1"]) -= -learning_rate * dParam;
    // cout << *(net.params["b1"]) << endl;
    // cout << dParam << endl;
    // *(net.params["b1"]) -= MatrixXd::Ones(1, hidden_size); // こちらは更新される->gradsによる更新がおかしい？
    // cout << *(net.params["b1"]) << endl;

    cout << "--- parameter b1 ---" << endl;
    cout << net.params["b1"] << endl;
    cout << "--- parameter b1 update ---" << endl;
    net.params["b1"] -= -learning_rate * grads["b1"];

    MatrixXd dParam = grads["b1"];
    net.params["b1"] -= -learning_rate * dParam;
    cout << net.params["b1"] << endl;
    cout << dParam << endl;
    net.params["b1"] -= MatrixXd::Ones(1, hidden_size); // こちらは更新される->gradsによる更新がおかしい？
    cout << net.params["b1"] << endl;

    return 0;
}