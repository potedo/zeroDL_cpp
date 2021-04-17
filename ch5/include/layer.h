#ifndef _LAYER_H_
#define _LAYER_H_

#include <Eigen/Dense>
#include <vector>

namespace MyDL
{

    using namespace Eigen;
    using std::vector;

    class BaseLayer
    {
    public:
        virtual vector<MatrixXd> forward(vector<MatrixXd>) = 0;
        virtual vector<MatrixXd> backward(vector<MatrixXd>) = 0;
    };

    class AddLayer : public BaseLayer
    {
        // この層に関しては、微分をそのまま流すので内部に順伝播の情報を保持する必要はない
    private:
    public:
        vector<MatrixXd> forward(vector<MatrixXd>);
        vector<MatrixXd> backward(vector<MatrixXd>);
    };

    class MulLayer : public BaseLayer
    {
    private:
        MatrixXd _x, _y;

    public:
        vector<MatrixXd> forward(vector<MatrixXd>);
        vector<MatrixXd> backward(vector<MatrixXd>);
    };

}

#endif // _LAYER_H_