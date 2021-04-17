#include <iostream>
#include <Eigen/Dense>
#include "./include/simple_layer.h"

using namespace Eigen;
using namespace MyDL;

int main(){
    using std::cout;
    using std::endl;

    MatrixXd apple = MatrixXd::Zero(1, 1);
    MatrixXd apple_num = MatrixXd::Zero(1, 1);
    MatrixXd orange = MatrixXd::Zero(1, 1);
    MatrixXd orange_num = MatrixXd::Zero(1, 1);
    MatrixXd tax = MatrixXd::Zero(1, 1);
    MatrixXd apple_price, orange_price, all_price, price;

    MatrixXd dprice = MatrixXd::Ones(1, 1);
    MatrixXd dapple_price, dorange_price, dall_price, dtax, dapple_num, dorange_num, dapple, dorange;

    apple(0) = 100;
    apple_num(0) = 2;
    orange(0) = 150;
    orange_num(0) = 3;
    tax(0) = 1.1;

    //layer
    MulLayer mul_apple_layer, mul_orange_layer, mul_tax_layer;
    AddLayer add_apple_orange_layer;

    // forward
    apple_price = mul_apple_layer.forward(apple, apple_num);
    orange_price = mul_orange_layer.forward(orange, orange_num);
    all_price = add_apple_orange_layer.forward(apple_price, orange_price);
    price = mul_tax_layer.forward(all_price, tax);

    // backward
    mul_tax_layer.backward(dprice, dall_price, dtax);
    add_apple_orange_layer.backward(dall_price, dapple_price, dorange_price);
    mul_orange_layer.backward(dorange_price, dorange, dorange_num);
    mul_apple_layer.backward(dapple_price, dapple, dapple_num);

    cout << "price: " << price << endl;
    cout << "dapple_num: " << dapple_num << endl;
    cout << "dapple: " << dapple << endl;
    cout << "dorange_num: " << dorange_num << endl;
    cout << "dorange: " << dorange << endl;
    cout << "dtax: " << dtax << endl;

    return 0;    
}