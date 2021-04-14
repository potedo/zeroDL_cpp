#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "../include/sample_network.h"

using namespace Eigen;

int main()
{
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using std::unordered_map;

    int input_size = 2;
    int output_size = 2;

    DebugAffine db_affine(input_size, output_size);

    // cout << "parameter W" << endl;
    // cout << *(db_affine.params["W"]) << endl;
    // cout << "parameter b" << endl;
    // cout << *(db_affine.params["b"]) << endl;

    // *(db_affine.params["W"]) += MatrixXd::Ones(input_size, output_size) * 0.5;
    // *(db_affine.params["b"]) += MatrixXd::Ones(1, output_size) * 0.5;

    // cout << "parameter W" << endl;
    // cout << *(db_affine.params["W"]) << endl;
    // cout << "parameter b" << endl;
    // cout << *(db_affine.params["b"]) << endl;

    cout << "parameter W" << endl;
    cout << db_affine.params["W"] << endl;
    cout << "parameter b" << endl;
    cout << db_affine.params["b"] << endl;
    cout << "Layer Params" << endl;
    db_affine.PrintLayerParams();

    db_affine.params["W"] += MatrixXd::Ones(input_size, output_size) * 0.5;
    db_affine.params["b"] += MatrixXd::Ones(1, output_size) * 0.5;

    cout << "parameter W" << endl;
    cout << db_affine.params["W"] << endl;
    cout << "parameter b" << endl;
    cout << db_affine.params["b"] << endl;
    cout << "Layer Params" << endl;
    db_affine.PrintLayerParams();

    return 0;
}