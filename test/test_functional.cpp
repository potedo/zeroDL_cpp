#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>

using namespace Eigen;
using std::vector;

double function(vector<MatrixXd>);
vector<MatrixXd> function2(vector<MatrixXd>);

int main()
{
    using std::cout;
    using std::endl;
    using std::vector;

    MatrixXd x = MatrixXd::Ones(2,2);
    MatrixXd W = MatrixXd::Ones(2,2);
    vector<MatrixXd> X;
    X.push_back(x);
    auto loss = [X](MatrixXd W) -> vector<MatrixXd> {return (function2(X));};
    std::function<vector<MatrixXd>(MatrixXd)> loss_W = loss;

    // double out;
    vector<MatrixXd> out;
    out = loss_W(W);
    cout << out[0] << endl;

    return 0;
}

double function(vector<MatrixXd> inputs){
    return inputs[0].sum();
}

vector<MatrixXd> function2(vector<MatrixXd> inputs)
{
    return inputs;
}