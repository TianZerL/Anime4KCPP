#pragma once

#include<sstream>
#include<exception>

#define TYPE_ITEM(Name)\
struct Name\
{\
    static constexpr const char* const string = #Name;\
};

namespace Anime4KCPP
{
    namespace ExceptionType
    {
        TYPE_ITEM(IO)
        TYPE_ITEM(RunTimeError)
        TYPE_ITEM(GPU)
    }

    template<typename errType, bool addtlInfo = false>
    class ACException;
}

template<typename errType>
class Anime4KCPP::ACException<errType, true> :public std::runtime_error
{
public:
    ACException(const std::string& errMsg, const int addltErrCode);
    ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode);
};

template<typename errType>
class Anime4KCPP::ACException<errType, false> :public std::runtime_error
{
public:
    ACException(const std::string& errMsg);
};

template<typename errType>
Anime4KCPP::ACException<errType, false>::ACException(const std::string& errMsg) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + errType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n"
    ) {}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const int addltErrCode) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + errType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n\n"
        "Additional error code :\n" 
        + std::to_string(addltErrCode)+"\n\n"
        "Additional information :\n"
        "No additional information\n"
    ) {}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode) :
    std::runtime_error(
        std::string(
            "An error occurred. \n\n"
            "Error type: ") + errType::string + "\n\n"
        "Error message :\n" +
        errMsg + "\n\n"
        "Additional error code :\n"
        + std::to_string(addltErrCode) + "\n\n"
        "Additional information :\n" +
        addtlInfo + "\n"
    ) {}
