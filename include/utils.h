#ifndef _UTILS_H_
#define _UTILS_H_

#include <picojson.h>
#include <string>

namespace MyDL{

    int load_json(std::string, picojson::object&);

}

#endif // _UTILS_H_