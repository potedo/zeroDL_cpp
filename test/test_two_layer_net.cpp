#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <Eigen/Dense>
#include "include/sample_network.h"

using namespace Eigen;

int main()
{
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::vector;
    using std::unordered_map;

    // I/O containers
    vector<MatrixXd> inputs;
    vector<MatrixXd> outputs;
    MatrixXd X = -2 * MatrixXd::Identity(2, 2) + MatrixXd::Ones(2, 2);
    MatrixXd t = MatrixXd::Identity(2, 2);
    inputs.push_back(X);

    // Define MLP(2-Layer)
    TwoLayerNet net(2, 3, 2, 0.01);

    cout << "--- predict ---" << endl;
    outputs = net.predict(inputs);
    cout << outputs[0] << endl;

    cout << "--- loss ---" << endl;
    inputs.push_back(t);
    outputs = net.loss(inputs);
    cout << outputs[0] << endl;

    cout << "--- accuracy ---" << endl;
    double accuracy = net.accuracy(inputs);
    cout << accuracy << endl;

    cout << "--- gradient ---" << endl;
    unordered_map<string, MatrixXd> grads;
    grads = net.gradient(inputs);

    for (auto grad: grads)
    {
        cout << grad.first << endl;
        cout << grad.second << endl;
    }

    // cout << "--- numerical gradient ---" << endl;
    // grads = net.numerical_gradient(inputs);
    
    // for (auto grad : grads)
    // {
    //     cout << grad.first << endl;
    //     cout << grad.second << endl;
    // }

    // 勾配更新確認
    for (int i = 0; i<5; i++){
        cout << *(net.params["b2"]) << endl;
        *(net.params["b2"]) -= grads["b2"];
        cout << "--- parameter b2 ---" << endl;
        cout << *(net.params["b2"]) << endl;
    }

    return 0;
}