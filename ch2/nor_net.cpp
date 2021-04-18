#include <iostream>
#include "../simple_lib/include/simple_perceptron.h"

int main()
{
    using std::cout;
    using std::endl;

    cout << "NOR(0, 0) = " << MyDL::NOR<double>(0, 0) << endl;
    cout << "NOR(1, 0) = " << MyDL::NOR<double>(1, 0) << endl;
    cout << "NOR(0, 1) = " << MyDL::NOR<double>(0, 1) << endl;
    cout << "NOR(1, 1) = " << MyDL::NOR<double>(1, 1) << endl;

    return 0;
}