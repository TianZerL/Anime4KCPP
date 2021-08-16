#pragma once

#include<string>
#include<stdexcept>

#define TYPE_ITEM(Name)\
struct Name\
{\
    static constexpr const char* string = #Name;\
};

namespace Anime4KCPP
{
    namespace ExceptionType
    {
        TYPE_ITEM(IO)
        TYPE_ITEM(RunTimeError)
        TYPE_ITEM(GPU)
    }

    template<typename exceptionType, bool addtlInfo = false>
    class ACException;
}

template<typename exceptionType>
class Anime4KCPP::ACException<exceptionType, true> :public std::runtime_error
{
public:
    ACException(const std::string& errMsg, int addltErrCode);
    ACException(const std::string& errMsg, const std::string& addtlInfo, int addltErrCode);
};

template<typename exceptionType>
class Anime4KCPP::ACException<exceptionType, false> :public std::runtime_error
{
public:
    ACException(const std::string& errMsg);
};

template<typename exceptionType>
Anime4KCPP::ACException<exceptionType, false>::ACException(const std::string& errMsg) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + exceptionType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n"
    ) {}

template<typename exceptionType>
Anime4KCPP::ACException<exceptionType, true>::ACException(const std::string& errMsg, const int addltErrCode) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + exceptionType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n\n"
        "Additional error code :\n" 
        + std::to_string(addltErrCode)+"\n\n"
        "Additional information :\n"
        "No additional information\n"
    ) {}

template<typename exceptionType>
Anime4KCPP::ACException<exceptionType, true>::ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + exceptionType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n\n"
        "Additional error code :\n"
        + std::to_string(addltErrCode) + "\n\n"
        "Additional information :\n" +
        addtlInfo + "\n"
    ) {}
