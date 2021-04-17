#include <vector>
#include <matplotlibcpp.h>
#include "../ch5/include/simple_activation.h"

namespace plt = matplotlibcpp;

int main(){
    using std::vector;

    int n = 100;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i)
    {
        x.push_back(i - 50);
        y.push_back(MyDL::step_function<double>(i - 50));
    }

    plt::plot(x, y, "--r");
    // plt::save("ch3/step_function_visualize_sample.png");
    plt::show();

    // plt::plot(y, x, "--b");
    // plt::save("ch3/step_function_visualize_sample2.png");

    return 0;
}