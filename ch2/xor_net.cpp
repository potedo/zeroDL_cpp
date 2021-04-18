#include <iostream>
#include "../simple_lib/include/simple_perceptron.h"

int main()
{
    using std::cout;
    using std::endl;

    cout << "XOR(0, 0) = " << MyDL::XOR<double>(0, 0) << endl;
    cout << "XOR(1, 0) = " << MyDL::XOR<double>(1, 0) << endl;
    cout << "XOR(0, 1) = " << MyDL::XOR<double>(0, 1) << endl;
    cout << "XOR(1, 1) = " << MyDL::XOR<double>(1, 1) << endl;

    return 0;
}