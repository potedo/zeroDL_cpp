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

    int LittleEndian2BigEndian(int);

    // --------------------------------------------
    //              シンプルなMNISTローダ
    // --------------------------------------------
    class Mnist{
        public:
            vector<vector<double>> readTrainingFile(string filename);
            vector<double> readLabelFile(string filename);
    };

    // ---------------------------------------------
    //              Eigen用 MNISTローダ
    // ---------------------------------------------
    class MnistEigenDataset
    {

        private:
            // ファイル入力用
            string _train_image_filepath = "./datasets/data/train-images.idx3-ubyte";
            string _train_label_filepath = "./datasets/data/train-labels.idx1-ubyte";
            string _test_image_filepath = "./datasets/data/t10k-images.idx3-ubyte";
            string _test_label_filepath = "./datasets/data/t10k-labels.idx1-ubyte";
            ifstream _train_image_ifs;
            ifstream _train_label_ifs;
            ifstream _test_image_ifs;
            ifstream _test_label_ifs;

            // ファイルシーク：初期位置記憶用
            ifstream::pos_type _train_image_pos;
            ifstream::pos_type _train_label_pos;
            ifstream::pos_type _test_image_pos;
            ifstream::pos_type _test_label_pos;

            // ファイルシーク用インデックス格納コンテナ
            vector<int> _train_indices;
            vector<int> _test_indices;

            // 内部変数
            int _batch_size;
            int _train_max_batch_num;
            int _test_max_batch_num;
            int _train_load_count = 0;
            int _test_load_count = 0;

            int _number_of_train_data = 0;
            int _number_of_test_data = 0;
            int _rows = 0;
            int _cols = 0;

        private:
            void _init_train_loader(void); // インデックスのシャッフルは初期化関数の外で行う
            void _init_test_loader(void);

        public:
            MnistEigenDataset(){}; // デフォルトコンストラクタ
            MnistEigenDataset(const int batch_size, bool random_load=true);
            void set_train_image_filepath(string);
            void set_train_label_filepath(string);
            void set_test_image_filepath(string);
            void set_test_label_filepath(string);
            void initialize_loader(void);
            void next_train(MatrixXd &, MatrixXd &, bool one_hot_label=false, bool normalize=true);
            void next_test(MatrixXd &, MatrixXd &, bool one_hot_label=false, bool normalize=true);

    };
}

#endif // _MNIST_H_