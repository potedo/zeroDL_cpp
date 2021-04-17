#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;

    MatrixXd m(2,2);
    m << 1, 2,
         3, 4;

    MatrixXd::Index maxRow, maxCol;
    double max = m.maxCoeff(&maxRow, &maxCol);

    cout << "max of m: " << max << " at (" << maxRow <<"," << maxCol << ")" << endl;

    return 0;
}