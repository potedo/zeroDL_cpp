#include <iostream>
#include <vector>
#include <matplotlibcpp.h>
#include "../simple_lib/include/simple_activation.h"
using namespace std;

namespace plt = matplotlibcpp;

int main()
{
    cout << "matplotlib-cpp sample start" << endl;

    int n = 100;
    vector<double> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i)
    {
        x.push_back(i-50);
        y.push_back(MyDL::step_function<double>(i-50));
        // z.push_back(cos(2 * M_PI * i / 360.0));
    }

    // plt::named_plot("step function", x, y, "--r");
    // plt::plot(x, z, ".-b");
    // plt::grid(true);
    // plt::legend();
    // plt::save("ch3/step_function_visualize1.png");
    // plt::show();
    // plt::pause(0.1);

    // for (int i = 0; i < n; ++i)
    // {
    //     x[i] = i -50;
    //     z[i] = (cos(2 * M_PI * i / 360.0));
    // }

    plt::named_plot("sin funciton", x, y, "--b");
    plt::grid(true);
    plt::legend(); // legendを入れるとバグらない傾向にある？ 
    plt::save("ch3/step_function_visualize_sample2.png");

    return 0;
}