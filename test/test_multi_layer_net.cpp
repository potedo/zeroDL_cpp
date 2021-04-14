#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <Eigen/Dense>
#include "../include/sample_network.h"

using namespace Eigen;

int main()
{
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::vector;
    using std::string;
    using std::unordered_map;

    int batch_size = 4;
    int input_size = 2;
    vector<int> hidden_size = {3};
    int output_size = 2;
    double lambda = 2.0;

    MultiLayerNet net(input_size, hidden_size, output_size, lambda);

    cout << "Check Initialized Parameters" << endl;
    for (auto param : net.params)
    {
        cout << param.first << endl;
        cout << *(param.second) << endl;
    }

    cout << "Test Predict Method" << endl;
    vector<MatrixXd> inputs, outputs;
    MatrixXd X = MatrixXd::Random(batch_size, input_size);
    inputs.push_back(X);

    outputs = net.predict(inputs);

    cout << outputs[0] << endl;


    cout << "Test Loss Method" << endl;
    vector<MatrixXd> loss_output;
    MatrixXd t = MatrixXd::Zero(batch_size, output_size);

    t << 1, 0,
         1, 0,
         1, 0,
         0, 1;

    loss_output = net.loss(inputs, t);

    cout << loss_output[0] << endl;

    cout << "Test Accuracy Method" << endl;
    double accuracy;

    accuracy = net.accuracy(inputs, t);

    cout << accuracy << endl;

    cout << "Test Gradient Method" << endl;

    unordered_map<string, MatrixXd> grads;

    grads = net.gradient(inputs, t);

    for (auto grad : grads)
    {
        cout << grad.first << endl;
        cout << grad.second << endl;
    }

    cout << "Test Finished" << endl;

    return 0;
}
