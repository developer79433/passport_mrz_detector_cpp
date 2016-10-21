#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "getcwd.h"

using namespace std;

string getcwd(void) {
    string result(1024, '\0');
    while (getcwd(&result[0], result.size()) == 0) {
        if( errno != ERANGE ) {
          throw runtime_error(strerror(errno));
        }
        result.resize(result.size() * 2);
    }
    result.resize(result.find('\0'));
    return result;
}
