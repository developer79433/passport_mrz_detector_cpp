#ifndef MRZ_H
#define MRZ_H 1

#include <cassert>

class MRZ {
public:
    enum mrz_type { UNKNOWN = 0, TYPE_1 = 1, UNUSED = 2, TYPE_3 = 3 };
    MRZ(void);
    static unsigned int getCharsPerLine(enum mrz_type type) {
        switch (type) {
        case TYPE_1:
            return 30;
        case TYPE_3:
            return 44;
        default:
            break;
        }
        return 0;
    };
    static unsigned int getLineCount(enum mrz_type type) {
        switch (type) {
        case TYPE_1:
            return 3;
        case TYPE_3:
            return 2;
        default:
            break;
        }
        return 0;
    };
    static const char *charset;
    static unsigned int getCharsPerLine(void) { assert(0); return 0; };
    static unsigned int getLineCount(void) { assert(0); return 0; };
};

class MRZType1 : public MRZ {
public:
    static unsigned int getCharsPerLine(void) { return 30; };
    static unsigned int getLineCount(void) { return 3; };
};

class MRZType3 : public MRZ {
public:
    static unsigned int getCharsPerLine(void) { return 44; };
    static unsigned int getLineCount(void) { return 2; };
};

#endif /* MRZ_H */
