#include <Eigen/Dense>
#include "../include/numerical_gradient.h"

namespace MyDL
{

    using namespace Eigen;

    MatrixXd numerical_gradient(double (*f)(MatrixXd&), MatrixXd& x)
    {
        double h = 1e-4;
        int size_x = x.rows() * x.cols();
        MatrixXd grad(x.rows(), x.cols());

        for (int i = 0; i < size_x; i++)
        {
            double tmp_val = x(i);
            double f_plus_h;
            double f_minus_h;

            // f(x + h)
            x(i) = tmp_val + h;
            f_plus_h = f(x);

            // f(x - h)
            x(i) = tmp_val - h;
            f_minus_h = f(x);

            grad(i) = (f_plus_h - f_minus_h) / (2 * h);

            x(i) = tmp_val;
        }

        return grad;
    }

    // TwoLayerNetなどで用いるもの(ラムダ式使用・DNNのインスタンスに含まれるパラメータを参照して使用する場合)
    MatrixXd numerical_gradient(const std::function<double(MatrixXd)> &f, MatrixXd &x)
    {
        double h = 1e-4;
        int size_x = x.rows() * x.cols();
        MatrixXd grad(x.rows(), x.cols());

        for (int i = 0; i < size_x; i++)
        {
            double tmp_val = x(i);
            double f_plus_h;
            double f_minus_h;

            // f(x + h)
            x(i) = tmp_val + h;
            f_plus_h = f(x);

            // f(x - h)
            x(i) = tmp_val - h;
            f_minus_h = f(x);

            grad(i) = (f_plus_h - f_minus_h) / (2 * h);

            x(i) = tmp_val;
        }

        return grad;
    }

    MatrixXd numerical_gradient(const std::function<vector<MatrixXd>(MatrixXd)> &f, MatrixXd &X)
    {
        double h = 1e-4;
        int size_X = X.rows() * X.cols();
        MatrixXd grad(X.rows(), X.cols());
        double f_plus_h, f_minus_h;

        for (int i = 0; i < size_X; i++)
        {
            double tmp_val = X(i);

            // f(x + h)
            X(i) = tmp_val + h;
            f_plus_h = f(X)[0].sum();

            // f(x - h)
            X(i) = tmp_val - h;
            f_minus_h = f(X)[0].sum();

            grad(i) = (f_plus_h - f_minus_h) / (2 * h);

            X(i) = tmp_val;
        }

        return grad;
    }

    MatrixXd numerical_gradient(const std::function<vector<MatrixXd>(VectorXd)> &f, VectorXd &X)
    {
        double h = 1e-4;
        int size_X = X.rows();
        VectorXd grad(X.rows());
        double f_plus_h, f_minus_h;

        for (int i = 0; i < size_X; i++)
        {
            double tmp_val = X(i);

            // f(x + h)
            X(i) = tmp_val + h;
            f_plus_h = f(X)[0].sum();

            // f(x - h)
            X(i) = tmp_val - h;
            f_minus_h = f(X)[0].sum();

            grad(i) = (f_plus_h - f_minus_h) / (2 * h);

            X(i) = tmp_val;
        }

        return grad;
    }
}