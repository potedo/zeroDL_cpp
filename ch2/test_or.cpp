#include <iostream>
#include "../include/simple_perceptron.h"

int main()
{
    using std::cout;
    using std::endl;

    cout << "OR(0, 0) = " << MyDL::OR<double>(0, 0) << endl;
    cout << "OR(1, 0) = " << MyDL::OR<double>(1, 0) << endl;
    cout << "OR(0, 1) = " << MyDL::OR<double>(0, 1) << endl;
    cout << "OR(1, 1) = " << MyDL::OR<double>(1, 1) << endl;

    return 0;
}