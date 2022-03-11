#ifndef _FILE_UTIL_H
#define _FILE_UTIL_H
#include <fstream>
#include "particle_set.h"

template<typename T>
inline void writeToFile(std::ostream& file, const T& a)
{
    const char* binaryData = reinterpret_cast<const char*>(&a);
    file.write(binaryData, sizeof(T));
}

inline void writeToFile(std::ostream& file, const vec2d_t& a)
{
    writeToFile(file, a.x);
    writeToFile(file, a.y);
}

template<typename T>
inline void readFromFile(std::istream& file, T& a)
{
    char* binaryData = reinterpret_cast<char*>(&a);
    file.read(binaryData, sizeof(T));
}

inline void readFromFile(std::istream& file, vec2d_t& a)
{
    readFromFile(file, a.x);
    readFromFile(file, a.y);
}

#endif