#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include <vector>

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(){
    using std::vector;

    int n = 5000;
    Eigen::VectorXd x(n), y(n), z(n), w = Eigen::VectorXd::Ones(n);
    for (int i = 0; i < n; ++i)
    {
        double value = (1.0 + i) / n;
        x(i) = value;
        y(i) = value * value;
        z(i) = value * value * value;
    }

    vector<double> xs(n), ys(n), zs(n), ws(n);

    Map<VectorXd>(&xs[0], n) = x;
    Map<VectorXd>(&ys[0], n) = y;
    Map<VectorXd>(&zs[0], n) = z;
    Map<VectorXd>(&ws[0], n) = w;

    plt::loglog(xs, ys);                             // f(x) = x^2
    plt::loglog(xs, ws, "r--");                      // f(x) = 1, red dashed line
    plt::loglog(xs, zs, "g:"); // f(x) = x^3, green dots + label

    plt::title("Some functions of $x$"); // add a title

    plt::save("sample_eigen2.png");

    return 0;
}