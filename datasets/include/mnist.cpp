#include "mnist.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

namespace MyDL{

    using std::ifstream;
    using std::string;
    using std::vector;
    using std::ios;
    using std::cout;
    using std::endl;
    using namespace Eigen;

    // リトルエンディアン → ビッグエンディアンへの変換
    int LittleEndian2BigEndian(int i)
    {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + ((int)c4);
    }

    vector<vector<double>> Mnist::readTrainingFile(string filename){
        ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary); // バイナリモード・読み込みオンリー
        int magic_number = 0; // 2051が入る → これでMnistの画像データかどうかを判断
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;

        ifs.read((char*)&magic_number, sizeof(magic_number)); // sizeof(magic_number) = 4, つまり4byte分読み出し
        magic_number = LittleEndian2BigEndian(magic_number);
        ifs.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = LittleEndian2BigEndian(number_of_images);
        ifs.read((char*)&rows, sizeof(rows));
        rows = LittleEndian2BigEndian(rows);
        ifs.read((char*)&cols, sizeof(cols));
        cols = LittleEndian2BigEndian(cols);

        vector<vector<double>> images(number_of_images);
        cout << "magic number: " << magic_number << endl; 
        cout << "number of images: " << number_of_images << endl;
        cout << "rows: " << rows << endl;
        cout << "cols: " << cols << endl;

        for(int i=0; i<number_of_images; i++){
            images[i].resize(rows*cols); // 画像1毎分のストリーム領域確保

            for(int row = 0; row < rows; row++){
                for(int col = 0; col < cols; col++){
                    unsigned char tmp = 0;
                    ifs.read((char*)&tmp, sizeof(tmp));
                    images[i][rows*row+col] = (double)tmp;
                }
            }
        }
        return images;
    }

    vector<double> Mnist::readLabelFile(string filename){
        ifstream ifs(filename, std::ios::in | std::ios::binary);
        int magic_number = 0;
        int number_of_images = 0;

        ifs.read((char*)&magic_number, sizeof(magic_number));
        magic_number = LittleEndian2BigEndian(magic_number);
        ifs.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = LittleEndian2BigEndian(number_of_images);

        vector<double> label(number_of_images);

        cout << "number of images: " << number_of_images << endl;

        for(int i=0; i<number_of_images; i++){ // i=0 のところが、iだけになっていたのでバグっていた(初期化していなくてもコンパイルできてしまう)
            unsigned char tmp;
            ifs.read((char*)&tmp, sizeof(tmp));
            label[i] = (double)tmp;
        }
        return label;
    }

    MnistEigenDataset::MnistEigenDataset(int batch_size)
    {
        _batch_size = batch_size;
        _train_max_batch_num = (60000 + _batch_size - 1) / _batch_size; // 切り上げ
        _test_max_batch_num = (10000 + _batch_size - 1) / _batch_size;
    }

    void MnistEigenDataset::check_private_variable(void){
        cout << "_batch_size: " << _batch_size << endl;
        cout << "_train_max_batch_num: " << _train_max_batch_num << endl;
        cout << "_test_max_batch_num: " << _test_max_batch_num << endl; 
        cout << "_train_load_count: " << _train_load_count << endl;
        cout << "_test_load_count: " << _test_load_count << endl;
    }

    // 【指定されたバッチサイズ分のデータをMatrixとして返す関数】
    // MatrixXd& train_X -> 出力：訓練画像データ(batch_size×(28*28))
    // MatrixXd& train_y -> 出力：訓練ラベルデータ(batch_size×1 or 10)
    // bool one_hot_label -> 入力：ラベルデータをonehot化するか否かのフラグ
    //
    // この関数はポリモーフィズムで読み出す形を色々変換しても良いかも → ランダムなインデックスで読み出すとか
    void MnistEigenDataset::next_train(MatrixXd& train_X, MatrixXd& train_y, bool one_hot_label){

        if (_train_load_count==0){
            if (!_train_image_ifs.is_open())
            {
                _train_image_ifs.open(_train_image_filepath, std::ios::in | std::ios::binary);
                _train_label_ifs.open(_train_label_filepath, std::ios::in | std::ios::binary);
                int magic_number = 0;
                int number_of_images = 0;
                int rows = 0;
                int cols = 0;

                // 適切な位置までディスクリプタ移動
                _train_image_ifs.read((char *)&magic_number, sizeof(magic_number)); // sizeof(magic_number) = 4, つまり4byte分読み出し
                magic_number = LittleEndian2BigEndian(magic_number);
                _train_image_ifs.read((char *)&number_of_images, sizeof(number_of_images));
                number_of_images = LittleEndian2BigEndian(number_of_images);
                _train_image_ifs.read((char *)&rows, sizeof(rows));
                rows = LittleEndian2BigEndian(rows);
                _train_image_ifs.read((char *)&cols, sizeof(cols));
                cols = LittleEndian2BigEndian(cols);

                cout << "IMAGE magic number: " << magic_number << endl;

                // 適切な位置までディスクリプタ移動
                _train_label_ifs.read((char *)&magic_number, sizeof(magic_number));
                magic_number = LittleEndian2BigEndian(magic_number);
                _train_label_ifs.read((char *)&number_of_images, sizeof(number_of_images));
                number_of_images = LittleEndian2BigEndian(number_of_images);

                cout << "LABEL magic number: " << magic_number << endl;

                // シーク位置の記憶
                _train_image_pos = _train_image_ifs.tellg();
                _train_label_pos = _train_label_ifs.tellg();
        
            } else {
                cout << "next_train call reset!!" << endl;

                _train_image_ifs.clear();
                _train_image_ifs.seekg(_train_image_pos);
                _train_label_ifs.clear();
                _train_label_ifs.seekg(_train_label_pos);
            }
        }  

        _train_load_count++;

        vector<double> tmp_image(28*28); // 配列のサイズ初期化が必要(でないとセグフォ)
        vector<double> tmp_labels(_batch_size);

        // ここに必要なだけ訓練データを読み出す処理を作成
        for (int i = 0; i < _batch_size; i++)
        {
            for (int j = 0; j < 28 * 28; j++)
            {
                unsigned char tmp_value;
                _train_image_ifs.read((char *)&tmp_value, sizeof(tmp_value));
                tmp_image[j] = (double)(tmp_value);
            }
            train_X.row(i) = Map<Matrix<double, 1, 28 * 28>>(&(tmp_image[0]));
        }

        if (one_hot_label){
            train_y = MatrixXd::Zero(_batch_size, 10);
            for (int i = 0; i < _batch_size; i++)
            {
                unsigned char tmp_label;
                _train_label_ifs.read((char *)&tmp_label, sizeof(tmp_label));
                train_y(i, int(tmp_label)) = 1;
            }
        }
        else
        {
            // batch_sizeの分だけラベルデータを読み出す処理 → onehotのフラグでサイズ切り替え
            for (int i = 0; i < _batch_size; i++)
            {
                unsigned char tmp_label;
                _train_label_ifs.read((char *)&tmp_label, sizeof(tmp_label));
                tmp_labels[i] = (double)(tmp_label);
                train_y.row(i) = Map<Matrix<double, 1, 1>>(&(tmp_labels[i]));
            }
        }

        if (_train_load_count == _train_max_batch_num)
        {
            _train_load_count = 0;
        }
    }

    void MnistEigenDataset::next_test(MatrixXd& test_X, MatrixXd& test_y, bool one_hot_label){
        
        if (_test_load_count == 0){
            if (!_test_image_ifs.is_open())
            {
                _test_image_ifs.open(_test_image_filepath, std::ios::in | std::ios::binary);
                _test_label_ifs.open(_test_label_filepath, std::ios::in | std::ios::binary);
                int magic_number = 0;
                int number_of_images = 0;
                int rows = 0;
                int cols = 0;

                // 適切な位置までディスクリプタ移動
                _test_image_ifs.read((char *)&magic_number, sizeof(magic_number)); // sizeof(magic_number) = 4, つまり4byte分読み出し
                magic_number = LittleEndian2BigEndian(magic_number);
                _test_image_ifs.read((char *)&number_of_images, sizeof(number_of_images));
                number_of_images = LittleEndian2BigEndian(number_of_images);
                _test_image_ifs.read((char *)&rows, sizeof(rows));
                rows = LittleEndian2BigEndian(rows);
                _test_image_ifs.read((char *)&cols, sizeof(cols));
                cols = LittleEndian2BigEndian(cols);

                cout << "IMAGE magic number: " << magic_number << endl;
                
                // 適切な位置までディスクリプタ移動
                _test_label_ifs.read((char *)&magic_number, sizeof(magic_number));
                magic_number = LittleEndian2BigEndian(magic_number);
                _test_label_ifs.read((char *)&number_of_images, sizeof(number_of_images));
                number_of_images = LittleEndian2BigEndian(number_of_images);

                cout << "LABEL magic number: " << magic_number << endl;
                
                // シーク位置の記憶
                _test_image_pos = _test_image_ifs.tellg();
                _test_label_pos = _test_label_ifs.tellg();
            
            } else {
                cout << "next_test call reset!!" << endl;

                _test_image_ifs.clear();
                _test_image_ifs.seekg(_test_image_pos); // ファイルポインタを先頭に戻す
                _test_label_ifs.clear();
                _test_label_ifs.seekg(_test_label_pos); // ファイルポインタを先頭に戻す
            }
        }

        _test_load_count++;

        vector<double> tmp_image(28 * 28); // 配列のサイズ初期化が必要(でないとセグフォ)
        vector<double> tmp_labels(_batch_size);

        // ここに必要なだけ訓練データを読み出す処理を作成
        for (int i = 0; i < _batch_size; i++)
        {
            for (int j = 0; j < 28 * 28; j++)
            {
                unsigned char tmp_value;
                _test_image_ifs.read((char *)&tmp_value, sizeof(tmp_value));
                tmp_image[j] = (double)(tmp_value);
            }
            test_X.row(i) = Map<Matrix<double, 1, 28 * 28>>(&(tmp_image[0]));
        }

        if (one_hot_label){
            test_y = MatrixXd::Zero(_batch_size, 10);
            for (int i = 0; i < _batch_size; i++)
            {
                unsigned char tmp_label;
                _test_label_ifs.read((char*)&tmp_label, sizeof(tmp_label));
                test_y(i, int(tmp_label)) = 1;
            }

        }else{
            // batch_sizeの分だけラベルデータを読み出す処理 → onehotのフラグでサイズ切り替え
            for (int i = 0; i < _batch_size; i++)
            {
                unsigned char tmp_label;
                _test_label_ifs.read((char *)&tmp_label, sizeof(tmp_label));
                tmp_labels[i] = (double)(tmp_label);
                test_y.row(i) = Map<Matrix<double, 1, 1>>(&(tmp_labels[i]));
            }
        }

        if (_test_load_count == _test_max_batch_num)
        {
            _test_load_count = 0;
        }
    }


    // やっぱり読み出しがおかしい？デバッグ用関数
    void MnistEigenDataset::print_ifs(){
        if(_debug_counter==0){
            if(_train_image_ifs.is_open()){
                _train_image_ifs.close();
            }
            cout << "train image filepath: " << _train_image_filepath << endl; 
            _train_image_ifs.open(_train_image_filepath.c_str(), std::ios::in | std::ios::binary);
            int magic_number = 0;
            int number_of_images = 0;
            int rows = 0;
            int cols = 0;

            // 適切な位置までディスクリプタ移動
            _train_image_ifs.read((char *)&magic_number, sizeof(magic_number)); // sizeof(magic_number) = 4, つまり4byte分読み出し
            magic_number = LittleEndian2BigEndian(magic_number);
            _train_image_ifs.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = LittleEndian2BigEndian(number_of_images);
            _train_image_ifs.read((char *)&rows, sizeof(rows));
            rows = LittleEndian2BigEndian(rows);
            _train_image_ifs.read((char *)&cols, sizeof(cols));
            cols = LittleEndian2BigEndian(cols);

            cout << "magic number: " << magic_number << endl;
            cout << "number of images: " << number_of_images << endl;
            cout << "rows: " << rows << endl;
            cout << "cols: " << cols << endl;

            cout << endl;

            cout << "train label filepath: " << _train_label_filepath << endl;
            _train_label_ifs.open(_train_label_filepath, std::ios::in | std::ios::binary);
            
            // 適切な位置までディスクリプタ移動
            _train_label_ifs.read((char *)&magic_number, sizeof(magic_number));
            magic_number = LittleEndian2BigEndian(magic_number);
            _train_label_ifs.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = LittleEndian2BigEndian(number_of_images);

            cout << "magic number: " << magic_number << endl;
            cout << "number of images: " << number_of_images << endl;
        }

        // File Openのチェック
        if (_train_image_ifs.fail())
        {
            cout << "File Open Failed!" << endl;
            return;
        }

        cout << "debug print " << _debug_counter << "th process start!" << endl;
        vector<double> tmp_vector;
        for(int i=0; i<28*28; i++){
            unsigned char tmp;
            _train_image_ifs.read((char*)&tmp, sizeof(tmp));
            tmp_vector.push_back((double)tmp);
        }
        for(int i=0; i<28*28; i++){
            cout << tmp_vector[i] << " ";
        }
        cout << endl;


        vector<double> tmp_label_vector;
        for(int i=0; i<10; i++){
            unsigned char tmp;
            _train_label_ifs.read((char*)&tmp, sizeof(tmp));
            tmp_label_vector.push_back((double)tmp);
        }

        for(int i=0; i<10; i++){
            cout << tmp_label_vector[i] << " ";
        }
        cout << endl;

        _debug_counter++;
    }

    void MnistEigenDataset::set_train_image_filepath(string filepath){
        _train_image_filepath = filepath;
    }

    void MnistEigenDataset::set_train_label_filepath(string filepath){
        _train_label_filepath = filepath;
    }

    void MnistEigenDataset::set_test_image_filepath(string filepath){
        _test_image_filepath = filepath;
    }

    void MnistEigenDataset::set_test_label_filepath(string filepath){
        _test_label_filepath = filepath;
    }

}

