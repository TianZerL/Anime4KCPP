#pragma once

#include<string>

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

    class ACBaseException;

    std::ostream& operator<< (std::ostream& stream, Anime4KCPP::ACBaseException& exception);

    template<typename errType, bool addtlInfo = false>
    class ACException;
}

class Anime4KCPP::ACBaseException
{
public:
    ACBaseException() = default;
    virtual std::string what() noexcept = 0;
};

template<typename errType>
class Anime4KCPP::ACException<errType, true> :public Anime4KCPP::ACBaseException
{
public:
    ACException(const std::string& errMsg, const int addltErrCode);
    ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode);

    virtual std::string what() noexcept override;
private:
    std::string errorMessage;
    std::string additionalInformation;
    int additionalErrorCode;
};

inline std::ostream& Anime4KCPP::operator<< (std::ostream& stream, Anime4KCPP::ACBaseException& exception)
{
    stream << exception.what();
    return stream;
}

template<typename errType>
class Anime4KCPP::ACException<errType, false> :public Anime4KCPP::ACBaseException
{
public:
    ACException(const std::string& errMsg);

    virtual std::string what() noexcept override;
private:
    std::string errorMessage;
};

template<typename errType>
Anime4KCPP::ACException<errType, false>::ACException(const std::string& errMsg) :
    errorMessage(errMsg) {}

template<typename errType>
std::string Anime4KCPP::ACException<errType, false>::what() noexcept
{
    std::ostringstream source;
    source
        << "Error Type: " << errType::string << std::endl
        << "Error message :" << std::endl
        << errorMessage;

    return source.str();
}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const int addltErrCode) :
    errorMessage(errMsg),  additionalErrorCode(addltErrCode), additionalInformation("No additional information") {}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode) :
    errorMessage(errMsg), additionalInformation(addtlInfo), additionalErrorCode(addltErrCode) {}

template<typename errType>
std::string Anime4KCPP::ACException<errType, true>::what() noexcept
{
    std::ostringstream source;
    source
        << "Error Type: " << errType::string << std::endl
        << "Error message :" << std::endl
        << errorMessage << std::endl
        << "Additional error code :" << std::endl
        << additionalErrorCode << std::endl
        << "Additional information :" << std::endl
        << additionalInformation;

    return source.str();
}
