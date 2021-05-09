#include "../include/utils.h"
#include <picojson.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>

namespace MyDL{

    int load_json(std::string filepath, picojson::object& obj)
    {
        // ファイルの読み込み
        std::ifstream ifs(filepath, std::ios::in);

        if(ifs.fail())
        {
            std::cerr << "failed to read " + filepath << std::endl;
            return 1;
        }
        const std::string json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();

        // 読み込んだファイルのパース
        picojson::value v;
        const std::string err = picojson::parse(v, json);

        if (!err.empty())
        {
            std::cerr << err << std::endl;
            return 2;
        }

        // 正常にパースできたら、objectに格納 
        obj = v.get<picojson::object>();

        return 0;
    }


    // MatrixXd im2col(MatrixXd input_data, int H, int W, int C, int filter_h, int filter_w, int stride, int pad)
    // {
    //     int out_h = (H + 2*pad - filter_h) / stride + 1; // 除算は商を取得(int同士の除算は切り捨て)
    //     int out_w = (W + 2*pad - filter_w) / stride + 1;
    //     // ここにパディング処理を書きたい(Eigenで良い書き方があるか？)


    // }

}