#include <Eigen/Dense>
#include "../include/loss.h"

namespace MyDL
{

    double cross_entropy_error(MatrixXd &y, MatrixXd &t)
    {
        int batch_size = y.rows();
        double loss = -(t.array() * y.array().log()).sum() / batch_size;
        return loss;
    }

}