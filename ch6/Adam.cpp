#include <iostream>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include "../include/numerical_gradient.h"
#include "../include/optimizer.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

double sample_function(MatrixXd &);
MatrixXd sample_function_gradient(MatrixXd &);

int main()
{
    using std::cout;
    using std::endl;
    using std::unordered_map;
    using std::vector;
    using namespace MyDL;

    vector<vector<double>> x, y, z;
    MatrixXd X = MatrixXd::Zero(1, 2);
    for (double i = -10; i <= 10; i += 0.5)
    {
        vector<double> x_row, y_row, z_row;
        for (double j = -10; j <= 10; j += 0.25)
        {
            X(0) = i;
            X(1) = j;
            x_row.push_back(i);
            y_row.push_back(j);
            z_row.push_back(sample_function(X));
        }
        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    plt::plot_surface(x, y, z);
    plt::save("SGD_function.png");
    plt::clf();

    x.clear();
    y.clear();
    vector<double> ux, uy, vx, vy;
    MatrixXd grad = MatrixXd::Zero(1, 2);
    for (double i = -10; i <= 10; i++)
    {
        for (double j = -5; j <= 5; j++)
        {
            ux.push_back(i);
            uy.push_back(j);
            X(0) = i;
            X(1) = j;
            grad = sample_function_gradient(X);
            vx.push_back((double)-grad(0));
            vy.push_back((double)-grad(1));
        }
    }

    Adam optimizer(0.3, 0.7, 0.9);
    unordered_map<string, MatrixXd> params, grads;
    int num_update = 30;
    vector<vector<double>> param_history(2, vector<double>(num_update, 0));
    X(0) = -6;
    X(1) = 2;
    params["X"] = X;
    param_history[0][0] = X(0);
    param_history[1][0] = X(1);

    for (int i = 0; i < num_update; i++)
    {
        cout << "iteration: " << i << endl;
        grads["X"] = sample_function_gradient(params["X"]);
        optimizer.update(params, grads);

        param_history[0][i + 1] = (double)params["X"](0);
        param_history[1][i + 1] = (double)params["X"](1);
    }

    plt::quiver(ux, uy, vx, vy);
    plt::save("SGD_function_gradient.png");

    plt::plot(param_history[0], param_history[1], {{"color", "red"}, {"marker", "o"}, {"linestyle", "--"}, {"label", "$f(x, y) = \\frac{1}{20}x^{2} + y^{2}$"}});
    plt::plot(vector<double>(1, 0), vector<double>(1, 0), "b+");
    plt::title("Adam: $f(x,y) = 0.05x^{2} + y^{2}$");
    plt::legend();
    plt::save("Adam_function_gradient_descent.png");
    plt::cla();

    return 0;
}

double sample_function(MatrixXd &x)
{
    return (1 / 20) * x(0) * x(0) + x(1) * x(1);
}

MatrixXd sample_function_gradient(MatrixXd &x)
{
    MatrixXd grad(1, 2);
    grad(0) = 0.1 * x(0);
    grad(1) = 2 * x(1);
    return grad;
}