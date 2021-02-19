#include <iostream>
#include <Eigen/Dense>
#include "../include/simple_activation.h"
#include <map>
#include <string>

using namespace Eigen;

// prototype declaration
std::map<std::string, MatrixXd> init_network(void);
MatrixXd forward(std::map<std::string, MatrixXd> network, MatrixXd x);

// main function
int main(){
    using std::cout;
    using std::endl;
    using std::map;
    using std::string;

    MatrixXd X  = MatrixXd::Zero(1, 2);
    MatrixXd W1 = MatrixXd::Zero(2, 3);
    MatrixXd B1 = MatrixXd::Zero(1, 3);
    MatrixXd A1 = MatrixXd::Zero(1, 3);
    MatrixXd Z1 = MatrixXd::Zero(1, 3);
    MatrixXd W2 = MatrixXd::Zero(3, 2);
    MatrixXd B2 = MatrixXd::Zero(1, 2);
    MatrixXd A2 = MatrixXd::Zero(1, 2);
    MatrixXd Z2 = MatrixXd::Zero(1, 2);
    MatrixXd W3 = MatrixXd::Zero(2, 2);
    MatrixXd B3 = MatrixXd::Zero(1, 2);
    MatrixXd A3 = MatrixXd::Zero(1, 2);
    MatrixXd Y  = MatrixXd::Zero(1, 2);

    X << 1.0, 0.5;
    W1 << 0.1, 0.3, 0.5,  0.2, 0.4, 0.6;
    B1 << 0.1, 0.2, 0.3;

    W2 << 0.1, 0.4,  0.2, 0.5,  0.3, 0.6;
    B2 << 0.1, 0.2;

    W3 << 0.1, 0.3,  0.2, 0.4;
    B3 << 0.1, 0.2;

    A1 = X * W1 + B1;

    cout << "A1 = " << A1 << endl;

    Z1 = A1.unaryExpr([](double p){return MyDL::sigmoid<double>(p);});

    cout << "Z1 = " << Z1 << endl;

    A2 = Z1 * W2 + B2;
    Z2 = A2.unaryExpr([] (double p){return MyDL::sigmoid<double>(p);});

    cout << "A2 = " << A2 << endl;
    cout << "Z2 = " << Z2 << endl;

    A3 = Z2 * W3 + B3;
    Y = A3.unaryExpr([] (double p){return MyDL::identity_function<double>(p);});

    cout << "A3 = " << A3 << endl;
    cout << "Y = " << Y << endl;

    // call network function -> last activation is Softmax function
    map <string, MatrixXd> network;
    network = init_network();
    Y = forward(network, X);

    cout << "Function ver: Y = " << Y << endl;

    return 0;
}


// Implementation
std::map<std::string, MatrixXd> init_network(void)
{

    using std::map;
    using std::string;

    map<string, MatrixXd> network;

    MatrixXd W1 = MatrixXd::Zero(2, 3);
    MatrixXd b1 = MatrixXd::Zero(1, 3);
    MatrixXd W2 = MatrixXd::Zero(3, 2);
    MatrixXd b2 = MatrixXd::Zero(1, 2);
    MatrixXd W3 = MatrixXd::Zero(2, 2);
    MatrixXd b3 = MatrixXd::Zero(1, 2);

    W1 << 0.1, 0.3, 0.5, 0.2, 0.4, 0.6;
    b1 << 0.1, 0.2, 0.3;
    W2 << 0.1, 0.4, 0.2, 0.5, 0.3, 0.6;
    b2 << 0.1, 0.2;
    W3 << 0.1, 0.3, 0.2, 0.4;
    b3 << 0.1, 0.2;

    network["W1"] = W1;
    network["b1"] = b1;
    network["W2"] = W2;
    network["b2"] = b2;
    network["W3"] = W3;
    network["b3"] = b3;

    return network;
}

MatrixXd forward(std::map<std::string, MatrixXd> network, MatrixXd x)
{
    using std::map;
    using std::string;

    MatrixXd W1 = MatrixXd::Zero(2, 3);
    MatrixXd b1 = MatrixXd::Zero(1, 3);
    MatrixXd W2 = MatrixXd::Zero(3, 2);
    MatrixXd b2 = MatrixXd::Zero(1, 2);
    MatrixXd W3 = MatrixXd::Zero(2, 2);
    MatrixXd b3 = MatrixXd::Zero(1, 2);

    MatrixXd a1 = MatrixXd::Zero(1, 3);
    MatrixXd z1 = MatrixXd::Zero(1, 3);
    MatrixXd a2 = MatrixXd::Zero(1, 2);
    MatrixXd z2 = MatrixXd::Zero(1, 2);
    MatrixXd a3 = MatrixXd::Zero(1, 2);
    MatrixXd y = MatrixXd::Zero(1, 2);

    W1 = network["W1"];
    b1 = network["b1"];
    W2 = network["W2"];
    b2 = network["b2"];
    W3 = network["W3"];
    b3 = network["b3"];

    a1 = x * W1 + b1;
    z1 = a1.unaryExpr([] (double p){return MyDL::sigmoid<double>(p);});
    a2 = z1 * W2 + b2;
    z2 = a2.unaryExpr([] (double p){return MyDL::sigmoid<double>(p);});
    a3 = z2 * W3 + b3;
    y  = MyDL::softmax(a3);

    return y;
}