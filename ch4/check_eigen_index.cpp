#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;
    using std::vector;

    MatrixXd X = MatrixXd::Zero(3,3);
    MatrixXd Y = MatrixXd::Zero(3,1);

    X << 1,2,3, 4,5,6, 7,8,9;
    Y << 2, 0, 1;

    vector<int> row_idx{0,1,2};
    vector<int> col_idx{2,0,1};

    for(int i=0; i<3; i++){
        cout << X(i, (int)Y(i)) << " ";
    }
    cout << endl;

    return 0;
}