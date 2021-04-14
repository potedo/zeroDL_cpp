#include <iostream>

class Singleton
{
    private:
        bool train_flg;

    private:
        Singleton(){};

    public:
        static Singleton& getInstance() {
            static Singleton singleton;
            singleton.set_flag(true);
            return singleton;
        }

        void set_flag(bool flg){
             train_flg = flg;
        }

        bool getFlag(void){
            return train_flg;
        }
};



int main()
{
    using std::cout;
    using std::endl;

    bool train_flg;

    train_flg = Singleton::getInstance().getFlag();

    cout << train_flg << endl;

    return 0;
}