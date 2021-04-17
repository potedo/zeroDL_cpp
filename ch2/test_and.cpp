#include <iostream>
#include "../include/simple_perceptron.h"

int main(){
    using std::cout;
    using std::endl;

    cout << "AND(0, 0) = " << MyDL::AND<double>(0, 0) << endl;
    cout << "AND(1, 0) = " << MyDL::AND<double>(1, 0) << endl;
    cout << "AND(0, 1) = " << MyDL::AND<double>(0, 1) << endl;
    cout << "AND(1, 1) = " << MyDL::AND<double>(1, 1) << endl;

    return 0;
}