#ifndef _SIMPLE_LOSS_H_
#define _SIMPLE_LOSS_H_

#include <Eigen/Dense>

namespace MyDL{
    using namespace Eigen;

    double cross_entropy_error(MatrixXd&, MatrixXd&);

}

#endif // _SIMPLE_LOSS_H_