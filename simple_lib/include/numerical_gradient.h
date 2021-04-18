#ifndef _NUMERICAL_GRADIENT_H_
#define _NUMERICAL_GRADIENT_H_

#include <functional>
#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;

    MatrixXd numerical_gradient(double (*f)(MatrixXd), MatrixXd);
    MatrixXd numerical_gradient(const std::function<double(MatrixXd)>&, MatrixXd&);

}
#endif // _NUMERICAL_GRADIENT_H_