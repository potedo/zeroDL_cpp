#include <iostream>
#include "../ch5/include/simple_perceptron.h"

int main(){
    using std::cout;
    using std::endl;

    cout << "NAND(0, 0) = " << MyDL::NAND<double>(0, 0) << endl;
    cout << "NAND(1, 0) = " << MyDL::NAND<double>(1, 0) << endl;
    cout << "NAND(0, 1) = " << MyDL::NAND<double>(0, 1) << endl;
    cout << "NAND(1, 1) = " << MyDL::NAND<double>(1, 1) << endl;

    return 0;
}