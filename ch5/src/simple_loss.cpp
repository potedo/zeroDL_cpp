#include <Eigen/Dense>
#include "simple_loss.h"

namespace MyDL{

    double cross_entropy_error(MatrixXd& y, MatrixXd& t){
        int batch_size = y.rows();
        double ret = (t.array() * y.array().log()).sum() / batch_size;
        return -ret;
    }

}