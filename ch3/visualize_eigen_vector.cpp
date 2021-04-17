#include <matplotlibcpp.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(){
    using std::cout;
    using std::endl;
    using std::vector;

    VectorXd x = Eigen::VectorXd::LinSpaced(200, 0, 6);
    VectorXd y;

    y = x.array().sin().exp().matrix();

    vector<double> x_vec(200), y_vec(200);

    // 直接EigenのVectorを扱えないので、STLのコンテナに移し替えて使う
    Map<VectorXd>(&x_vec[0], 200) = x;
    Map<VectorXd>(&y_vec[0], 200) = y;

    plt::named_plot("sample", x_vec, y_vec, "--b");
    plt::grid(true);
    plt::legend();
    plt::save("./ch3/sample_eigen.png");

    return 0;
}