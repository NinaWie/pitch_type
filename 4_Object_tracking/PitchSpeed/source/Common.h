/*
 * COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:
 *
 * 
 * For the purposes of this copyright and license, "Contributing Authors"
 * is defined as the following set of individuals:
 *
 *    Carlos Augusto Dietrich (cadietrich@gmail.com)
 *
 * This library is supplied "AS IS".  The Contributing Authors disclaim 
 * all warranties, expressed or implied, including, without limitation, 
 * the warranties of merchantability and of fitness for any purpose. 
 * The Contributing Authors assume no liability for direct, indirect, 
 * incidental, special, exemplary, or consequential damages, which may 
 * result from the use of the this library, even if advised of the 
 * possibility of such damage.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * source code, or portions hereof, for any purpose, without fee, subject
 * to the following restrictions:
 *
 * 1. The origin of this source code must not be misrepresented.
 *
 * 2. Altered versions must be plainly marked as such and must not be 
 *    misrepresented as being the original source.
 *
 * 3. This Copyright notice may not be removed or altered from any source 
 *    or altered source distribution.
 *
 * The Contributing Authors specifically permit, without fee, and 
 * encourage the use of this source code as a component in commercial 
 * products. If you use this source code in a product, acknowledgment 
 * is not required but would be appreciated.
 *
 * 
 * "Software is a process, it's never finished, it's always evolving. 
 * That's its nature. We know our software sucks. But it's shipping! 
 * Next time we'll do better, but even then it will be shitty. 
 * The only software that's perfect is one you're dreaming about. 
 * Real software crashes, loses data, is hard to learn and hard to use. 
 * But it's a process. We'll make it less shitty. Just watch!"
 */

#if !defined(COMMON_INCLUDED)
#define COMMON_INCLUDED

#include <string>
#include <cfloat>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#if !defined(UINT8)
typedef unsigned char UINT8;

#define MAX_UINT8 ((UINT8)~((UINT8)0))
#endif //#if !defined(INT8)

#if !defined(UINT16)
typedef unsigned short UINT16;

#define MAX_UINT16 ((UINT16)~((UINT16)0))
#endif //#if !defined(UINT16)

#if !defined(UINT32)
typedef unsigned int UINT32;

#define MAX_UINT32 ((UINT32)~((UINT32)0))
#endif //#if !defined(UINT32)

#if !defined(UINT64)
// BUG: (11-Jan-2015) There is no __int64 on Apple OSX.
// BUG: (25-Nov-2015) Thre is an INT64 at /X11/Xmd.h, included by glew.
#if defined( __APPLE__) || defined(__linux)
//typedef uint64_t UINT64;
typedef unsigned long int UINT64;
#else
typedef unsigned __int64 UINT64;
#endif

#define MAX_UINT64 ((UINT64)~((UINT64)0))

#endif //#if !defined(UINT64)

#if !defined(INT8)
typedef signed char INT8;

#define MAX_INT8 ((INT8)(MAX_UINT8 >> 1))
#define MIN_INT8 ((INT8)~MAX_INT8)
#endif //#if !defined(INT8)

#if !defined(INT16)
typedef signed short INT16;

#define MAX_INT16 ((INT16)(MAX_UINT16 >> 1))
#define MIN_INT16 ((INT16)~MAX_INT16)
#endif //#if !defined(INT8)

#if !defined(INT32)
typedef signed int INT32;

#define MAX_INT32 ((INT32)(MAX_UINT32 >> 1))
#define MIN_INT32 ((INT32)~MAX_INT32)
#endif //#if !defined(INT8)

#if !defined(INT64)
// BUG: (11-Jan-2015) There is no __int64 on Apple OSX.
// BUG: (25-Nov-2015) Thre is an INT64 at /X11/Xmd.h, included by glew.
#if defined( __APPLE__) || defined(__linux)
//typedef int64_t INT64;
typedef long int INT64;
#else
typedef signed __int64 INT64;
#endif

#define MAX_INT64 ((INT64)(MAX_UINT64 >> 1))
#define MIN_INT64 ((INT64)~MAX_INT64)
#endif //#if !defined(INT8)

#if defined(WIN64)
#define AINT INT64
#define AUINT UINT64
#else //#if defined(WIN64)
#define AINT INT32
#define AUINT UINT32
#endif //#if defined(WIN64)

#if !defined(MIN)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif // #if !defined(MIN)

#if !defined(MAX)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif // #if !defined(MAX)

#if !defined(SORTED_PAIR)
#define SORTED_PAIR std::pair<int, int>
#endif // #if !defined(SORTED_PAIR)

#if !defined(MAKE_SORTED_PAIR)
#define MAKE_SORTED_PAIR(a, b) std::make_pair(MIN(a, b), MAX(a, b))
#endif // #if !defined(MAKE_SORTED_PAIR)

#if defined(_MSC_VER)
#define PORTABLE_FOPEN(fileStream, fileName, mode, returnValueIfFailed) if (fopen_s(&fileStream, fileName, mode) != 0)\
    {\
        /*CLogger::Instance().PushError(__FILE__, __LINE__);*/\
        return returnValueIfFailed;\
    }

#define PORTABLE_FSEEK(stream, offset, origin) _fseeki64(stream, (INT64)offset, origin)
#define PORTABLE_FTELL(stream) _ftelli64(stream)

#else //#if defined(_MSC_VER)
#define PORTABLE_FOPEN(fileStream, fileName, mode, returnValueIfFailed)\
    {\
        fileStream = fopen(fileName, mode);\
        if (fileStream == NULL)\
        {\
            CLogger::Instance().PushError((char *)__FILE__, __LINE__);\
            return returnValueIfFailed;\
        }\
    }

#define PORTABLE_FSEEK(stream, offset, origin) fseek(stream, offset, origin)
#define PORTABLE_FTELL(stream) ftell(stream)

#endif //#if defined(_MSC_VER)

/**
*/
template <typename T>
inline void ReverseByteOrdering16(T& value)
{
    unsigned char bytes[2] = {0};

    bytes[0] = (value >>  8) & 0xff;
    bytes[1] =  value        & 0xff;

    value = 0;

    value = (bytes[1]<<8)|bytes[0];
}

/**
*/
template <typename T>
void ReverseByteOrdering24(T& value)
{
    unsigned char bytes[3] = {0};

    bytes[0] = (value >> 16) & 0xff;
    bytes[1] = (value >>  8) & 0xff;
    bytes[2] =  value        & 0xff;

    value = 0;

    value = (bytes[2]<<16)|(bytes[1]<<8)|bytes[0];
}

// INT8 WRITE
#define FWRITE_INT8(fileStream, value)\
{\
    if (value >= MAX_INT8)\
    {\
        LOG_ERROR();\
        fclose(fileStream);\
        return false;\
    }\
    INT8 int8Buffer = (INT8)value;\
    fwrite(&int8Buffer, sizeof(INT8), 1, fileStream);\
}
// INT8 READ
#define FREAD_INT8(fileBuffer, int8Buffer)\
    int8Buffer = *((INT8*)(fileBuffer));\
    fileBuffer += sizeof(INT8);

// INT16 WRITE
#define FWRITE_INT16(fileStream, value)\
{\
    if (value >= MAX_INT16)\
    {\
        LOG_ERROR();\
        fclose(fileStream);\
        return false;\
    }\
    INT16 int16Buffer = (INT16)value;\
    fwrite(&int16Buffer, sizeof(INT16), 1, fileStream);\
}
// INT16 READ
#define FREAD_INT16(fileBuffer, int16Buffer)\
    int16Buffer = *((INT16*)(fileBuffer));\
    fileBuffer += sizeof(INT16);

// INT32 WRITE
#define FWRITE_INT32(fileStream, value)\
{\
    if (value >= MAX_INT32)\
    {\
        LOG_ERROR();\
        fclose(fileStream);\
        return false;\
    }\
    INT32 int32Buffer = (INT32)value;\
    fwrite(&int32Buffer, sizeof(INT32), 1, fileStream);\
}
// INT32 READ
#define FREAD_INT32(fileBuffer, int32Buffer)\
    int32Buffer = *((INT32*)(fileBuffer));\
    fileBuffer += sizeof(INT32);

// FLOAT32 WRITE
#define FWRITE_FLOAT32(fileStream, value)\
    {\
        float float32Buffer = (float)value;\
        fwrite(&float32Buffer, sizeof(float), 1, fileStream);\
    }
// FLOAT32 READ
#define FREAD_FLOAT32(fileBuffer, float32Buffer)\
    float32Buffer = *((float*)(fileBuffer));\
    fileBuffer += sizeof(float);

// WRITE STD::STRING
#define FWRITE_STD_STRING_8(fileStream, value)\
    {\
        if (value.size() > MAX_INT8)\
        {\
            LOG_ERROR();\
            fclose(fileStream);\
            return false;\
        }\
        INT8 int8Buffer = (INT8)value.size();\
        fwrite(&int8Buffer, sizeof(INT8), 1, fileStream);\
        if (int8Buffer > 0)\
            fwrite(value.c_str(), sizeof(char), int8Buffer, fileStream);\
    }
// READ STD::STRING
#define FREAD_CHAR_8(fileBuffer, buffer)\
    {\
        INT8 int8Buffer = *((INT8*)(fileStreamPointer));\
        fileBuffer += sizeof(INT8);\
        std::fill_n(buffer, 256, 0);\
        if (int8Buffer > 0)\
            memcpy(buffer, fileBuffer, int8Buffer);\
        fileBuffer += int8Buffer;\
    }

#define INCREASE_AS_CIRCULAR_ITERATOR(randomAccessIterator, container)  ++randomAccessIterator;\
    if (randomAccessIterator == container.end()) {\
        randomAccessIterator = container.begin(); \
        }

#define DECREASE_AS_CIRCULAR_ITERATOR(randomAccessIterator, container) if (randomAccessIterator == container.begin()) {\
        randomAccessIterator = container.end(); \
    }\
    --randomAccessIterator;

#define INCREASE_AS_ARRAY_ITERATOR(randomAccessIterator, container)  ++randomAccessIterator;\
    if (randomAccessIterator == container.end()) {\
        --randomAccessIterator; \
        }

#define DECREASE_AS_ARRAY_ITERATOR(randomAccessIterator, container) if (randomAccessIterator != container.begin()) {\
        --randomAccessIterator; \
    }

template < typename BidirectionalIterator, typename Container >
BidirectionalIterator NEXT_IN_ARRAY(BidirectionalIterator it, const Container& c)
{
    BidirectionalIterator nextIt = it;
    ++nextIt;
    if (nextIt == c.end())
        --nextIt;
    return nextIt;
}

template < typename BidirectionalIterator, typename Container >
BidirectionalIterator PREV_IN_ARRAY(BidirectionalIterator it, const Container& c)
{
    BidirectionalIterator prevIt = it;
    if (prevIt != c.begin())
        --prevIt;
    return prevIt;
}

#include <iostream>

// TESTING: (28-Apr-2015)
#define SINGLETON_DECLARATION(TYPE) \
public: \
    static TYPE& Instance() \
    { \
        if (!m_instance) \
        { \
            if (m_isDestroyed) \
                OnDeadReference(); \
            else \
                OnCreate(); \
        } \
        return *m_instance; \
    } \
private: \
    TYPE(); \
    TYPE(const TYPE&) = delete; \
    TYPE& operator=(const TYPE&) = delete; \
    static void OnCreate(); \
    static void OnDeadReference(); \
protected: \
    static TYPE* m_instance; \
    static bool m_isDestroyed; \
private:

// TESTING: (28-Apr-2015)
#define SINGLETON_DEFINITION(TYPE) \
TYPE* TYPE::m_instance = 0; \
bool TYPE::m_isDestroyed = false; \
void TYPE::OnCreate() \
{ \
    static TYPE theInstance; \
    m_instance = &theInstance; \
} \
void TYPE::OnDeadReference() \
{ \
    throw std::runtime_error("A dead reference to a singleton was detected."); \
}

namespace my {
    typedef unsigned char uint8;
    typedef unsigned short uint16;
    typedef unsigned int uint32;

    // BUG: (11-Jan-2015) There is no __int64 on Apple OSX.
    // BUG: (25-Nov-2015) Thre is an INT64 at /X11/Xmd.h, included by glew.
#if defined( __APPLE__) || defined(__linux)
    //typedef uint64_t UINT64;
    typedef unsigned long int uint64;
#else
    typedef unsigned __int64 uint64;
#endif

    typedef signed char int8;
    typedef signed short int16;
    typedef signed int int32;

    // BUG: (11-Jan-2015) There is no __int64 on Apple OSX.
    // BUG: (25-Nov-2015) Thre is an INT64 at /X11/Xmd.h, included by glew.
#if defined( __APPLE__) || defined(__linux)
    //typedef int64_t int64;
    typedef long int int64;
#else
    typedef signed __int64 int64;
#endif

    template <typename T>
    T Null();

    template <>
    inline my::int8 Null()
    {
        return MIN_INT8;
    }

    template <>
    inline my::int16 Null()
    {
        return MIN_INT16;
    }

    template <>
    inline INT32 Null()
    {
        return MIN_INT32;
    }

    template <>
    inline INT64 Null()
    {
        return MIN_INT64;
    }

    template <>
    inline float Null()
    {
        return FLT_MIN;
    }

    template <>
    inline double Null()
    {
        return DBL_MIN;
    }

    template <>
    inline std::string Null()
    {
        return "null";
    }

    template <typename T>
    inline bool IsNull(T value)
    {
        return value == Null<T>();
    }

    /**
    */
    inline std::string AddTrailingSlash(std::string path)
    {
        if (!path.empty())
        {
            if (path.find("/") != std::string::npos)
            {
                if (path.back() != '/')
                    path += '/';
            }
            else if (path.find("\\") != std::string::npos)
            {
                if (path.back() != '\\')
                    path += '\\';
            }
            else
                path += '/';
        }

        return path;
    }

    /**
    */
    inline std::string GetFileName(std::string path)
    {
        std::string fileName = my::Null<std::string>();

        if (!path.empty())
        {
            std::string::size_type delimiter = path.find_last_of("\\/");

            if (delimiter != std::string::npos)
                fileName = path.substr(delimiter + 1);
            else
                fileName = path;
        }

        return fileName;
    }

    /**
    */
    inline std::string GetDirectory(std::string path)
    {
        std::string fileName = my::Null<std::string>();

        if (!path.empty())
        {
            std::string::size_type delimiter = path.find_last_of("\\/");

            if (delimiter != std::string::npos)
                fileName = path.substr(0, delimiter + 1);
        }

        return fileName;
    }

    /**
    */
    inline std::vector<std::string> GetDirectoryNameArray(std::string path)
    {
        std::vector<std::string> directoryNameArray;

        while (!path.empty())
        {
            std::string::size_type delimiter = path.find_first_of("\\/");

            if (delimiter != std::string::npos)
            {
                directoryNameArray.push_back(path.substr(0, delimiter));

                path = path.substr(delimiter + 1);
            }
            else
                path.clear();
        }

        return directoryNameArray;
    }

    /**
    */
    inline bool CreateDirectory(std::string path)
    {
        std::vector<std::string> directoryNameArray = my::GetDirectoryNameArray(path);

        std::string partialDirectoryName;

        for (std::vector<std::string>::const_iterator directoryNameIterator = directoryNameArray.begin(); directoryNameIterator != directoryNameArray.end(); ++directoryNameIterator)
        {
            partialDirectoryName += my::AddTrailingSlash(*directoryNameIterator);

            if (!boost::filesystem::is_directory(partialDirectoryName))
                boost::filesystem::create_directory(partialDirectoryName);
        }

        return boost::filesystem::is_directory(path);
    }

    /**
    */
    inline std::string GetFileExtension(std::string path)
    {
        std::string extension = my::Null<std::string>();

        if (!path.empty())
        {
            std::string::size_type delimiter = path.find_last_of(".");

            // (BEGIN OF) BUG: (18-Aug-2016) "/"
            if (delimiter != std::string::npos)
            {
                std::string::size_type otherDelimiter = path.find_last_of("/");

                if ((otherDelimiter != std::string::npos) &&
                    (otherDelimiter > delimiter))
                {
                    delimiter = std::string::npos;
                }
            }
            // (END OF) BUG: (18-Aug-2016) "/"

            if (delimiter != std::string::npos)
            {
                extension = path.substr(delimiter + 1);

                boost::algorithm::to_lower(extension);
            }
        }

        return extension;
    }

    /**
    */
    template <typename T>
    T Rand();

    template <>
    inline float Rand()
    {
        return (float)rand() / RAND_MAX;
    }

    template <>
    inline double Rand()
    {
        return (double)rand() / RAND_MAX;
    }

    inline bool Toss()
    {
        return Rand<float>() > 0.5f;
    }

    template < typename ValueType >
    std::string NumberToString(ValueType value, int numberOfDecimalPlaces = -1)
    {
        std::string numberString;

        try
        {
            numberString = boost::lexical_cast<std::string>(value);

            if ((numberOfDecimalPlaces != -1) &&
                (numberString.find(".") != std::string::npos) &&
                // BUG: (11-Oct-2016) DO NOT CUT NUMBERS ON SCIENTIFIC NOTATION!
                (numberString.find("e") == std::string::npos))
            {
                numberString = numberString.substr(0, std::min(numberString.find(".") + 1 + numberOfDecimalPlaces, numberString.size()));

                while (numberString.back() == '0')
                    numberString.pop_back();
            }

            if (numberString.back() == '.')
                numberString.pop_back();
        }
        catch (...)
        {
            numberString = my::Null<std::string>();
        }

        return numberString;
    }

    template < typename ValueType >
    ValueType StringToNumber(std::string valueString)
    {
        ValueType value = my::Null<ValueType>();

        try
        {
            value = boost::lexical_cast<ValueType>(valueString);
        }
        catch (...)
        {
            value = my::Null<ValueType>();
        }

        return value;
    }

    inline std::string ReplaceKeyword(std::string inputString, std::string keyword, std::string value)
    {
        std::string outputString = inputString;

        size_t index = 0;

        index = outputString.find(keyword, index);

        while (index != std::string::npos)
        {
            outputString.erase(index, keyword.length());

            outputString.insert(index, value);

            index = outputString.find(keyword, index);
        }

        return outputString;
    }
};

#if defined(ENABLE_CHAOS_MONKEY_TESTING)
#define CHAOS_MONKEY_TESTING(VALID_RETURN, FALSE_RETURN) ((my::Rand<double>() > 0.05) ? (VALID_RETURN) : (FALSE_RETURN))
#else //#if defined(ENABLE_CHAOS_MONKEY_TESTING)
#define CHAOS_MONKEY_TESTING(VALID_RETURN, FALSE_RETURN) (VALID_RETURN)
#endif //#if defined(ENABLE_CHAOS_MONKEY_TESTING)

#endif // #if !defined(COMMON_INCLUDED)

