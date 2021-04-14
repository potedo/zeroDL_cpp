#ifndef _LOSS_H_
#define _LOSS_H_

#include <Eigen/Dense>

namespace MyDL
{
    using namespace Eigen;

    double cross_entropy_error(MatrixXd &, MatrixXd &);

}

#endif // _LOSS_H_