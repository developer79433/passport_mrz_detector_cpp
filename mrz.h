#ifndef MRZ_H
#define MRZ_H 1

#include <cassert>
#include <string>

class MRZ {
public:
    MRZ(void) {};
    virtual ~MRZ() {};
    static unsigned int getMinCharsPerLine(void) { return 30; }
    static unsigned int getMaxCharsPerLine(void) { return 44; }
    static unsigned int getMinLineCount(void) { return 2; }
    static unsigned int getMaxLineCount(void) { return 3; }
	virtual unsigned int getCharsPerLine(void) const = 0;
	virtual unsigned int getLineCount(void) const = 0;
    static const std::string charset;
};

class MRZType1 : public MRZ {
public:
	virtual unsigned int getCharsPerLine(void) const { return 30; };
	virtual unsigned int getLineCount(void) const { return 3; };
};

class MRZType2 : public MRZ {
public:
	virtual unsigned int getCharsPerLine(void) const { return 36; };
	virtual unsigned int getLineCount(void) const { return 2; };
};

class MRZType3 : public MRZ {
public:
	virtual unsigned int getCharsPerLine(void) const { return 44; };
	virtual unsigned int getLineCount(void) const { return 2; };
};

#endif /* MRZ_H */
