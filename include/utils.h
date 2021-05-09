#ifndef _UTILS_H_
#define _UTILS_H_

#include <picojson.h>
#include <string>
#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;

    int load_json(std::string, picojson::object&);

    // MatrixXd im2col(MatrixXd, int, int, int, int, int, int, int);

}

#endif // _UTILS_H_