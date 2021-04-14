#ifndef _NUMERICAL_GRADIENT_H_
#define _NUMERICAL_GRADIENT_H_

#include <vector>
#include <functional>
#include <Eigen/Dense>

namespace MyDL
{

    using namespace Eigen;
    using std::vector;

    MatrixXd numerical_gradient(double (*f)(MatrixXd&), MatrixXd&);
    MatrixXd numerical_gradient(const std::function<double(MatrixXd)> &, MatrixXd &);
    MatrixXd numerical_gradient(const std::function<vector<MatrixXd>(MatrixXd)> &, MatrixXd &);
    MatrixXd numerical_gradient(const std::function<vector<MatrixXd>(VectorXd)> &, VectorXd &);
}
#endif // _NUMERICAL_GRADIENT_H_