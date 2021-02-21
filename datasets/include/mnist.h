#ifndef _MNIST_H_
#define _MNIST_H_

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

namespace MyDL{

    using std::string;
    using std::vector;
    using std::ifstream;
    using namespace Eigen;

    // typedef Matrix<double, 60000, 28 * 28, RowMajor> TrainImageMatrix;
    // typedef Matrix<double, 60000, Dynamic, RowMajor> TrainLabelMatrix;
    // typedef Matrix<double, 10000, 28 * 28, RowMajor> TestImageMatrix;
    // typedef Matrix<double, 10000, Dynamic, RowMajor> TestLabelMatrix;

    int LittleEndian2BigEndian(int);

    // シンプルなMNISTローダ
    class Mnist{
        public:
            vector<vector<double>> readTrainingFile(string filename);
            vector<double> readLabelFile(string filename);
    };


    // Eigenを用いたDNN学習用MNISTローダ
    class MnistEigenDataset
    {

        private:
            string _train_image_filepath = "./datasets/data/train-images.idx3-ubyte";
            string _train_label_filepath = "./datasets/data/train-labels.idx1-ubyte";
            string _test_image_filepath = "./datasets/data/t10k-images.idx3-ubyte";
            string _test_label_filepath = "./datasets/data/t10k-labels.idx1-ubyte";
            ifstream _train_image_ifs;
            ifstream _train_label_ifs;
            ifstream _test_image_ifs;
            ifstream _test_label_ifs;
            ifstream::pos_type _test_image_pos;
            ifstream::pos_type _test_label_pos;
            ifstream::pos_type _train_image_pos;
            ifstream::pos_type _train_label_pos;
            int _batch_size;
            int _train_max_batch_num;
            int _test_max_batch_num;
            int _train_load_count = 0;
            int _test_load_count = 0;

            int _debug_counter = 0;

        public:
            MnistEigenDataset(){}; // デフォルトコンストラクタ
            MnistEigenDataset(const int batch_size);
            void set_train_image_filepath(string);
            void set_train_label_filepath(string);
            void set_test_image_filepath(string);
            void set_test_label_filepath(string);
            void next_train(MatrixXd &, MatrixXd &, bool one_hot_label=false);
            void next_test(MatrixXd &, MatrixXd &, bool one_hot_label=false);
            void check_private_variable(void);

            void print_ifs(void); // デバッグ用メソッド

    };
}

#endif // _MNIST_H_