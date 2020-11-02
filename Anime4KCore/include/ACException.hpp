#pragma once

#include<sstream>

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

    std::ostream& operator<< (std::ostream& stream, const Anime4KCPP::ACBaseException& exception);

    template<typename errType, bool addtlInfo = false>
    class ACException;
}

class Anime4KCPP::ACBaseException
{
public:
    virtual std::string what() const noexcept = 0;
};

template<typename errType>
class Anime4KCPP::ACException<errType, true> :public Anime4KCPP::ACBaseException
{
public:
    ACException(const std::string& errMsg, const int addltErrCode);
    ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode);

    virtual std::string what() const noexcept override;
private:
    std::string errorMessage;
    std::string additionalInformation;
    int additionalErrorCode;
};

inline std::ostream& Anime4KCPP::operator<< (std::ostream& stream, const Anime4KCPP::ACBaseException& exception)
{
    stream << exception.what();
    return stream;
}

template<typename errType>
class Anime4KCPP::ACException<errType, false> :public Anime4KCPP::ACBaseException
{
public:
    ACException(const std::string& errMsg);

    virtual std::string what() const noexcept override;
private:
    std::string errorMessage;
};

template<typename errType>
Anime4KCPP::ACException<errType, false>::ACException(const std::string& errMsg) :
    errorMessage(errMsg) {}

template<typename errType>
std::string Anime4KCPP::ACException<errType, false>::what() const noexcept
{
    std::ostringstream source;
    source
        << "An error occurred. " << std::endl
        << std::endl
        << "Error type: " << errType::string << std::endl
        << std::endl
        << "Error message :" << std::endl
        << errorMessage << std::endl;

    return source.str();
}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const int addltErrCode) :
    errorMessage(errMsg),  additionalErrorCode(addltErrCode), additionalInformation("No additional information") {}

template<typename errType>
Anime4KCPP::ACException<errType, true>::ACException(const std::string& errMsg, const std::string& addtlInfo, const int addltErrCode) :
    errorMessage(errMsg), additionalInformation(addtlInfo), additionalErrorCode(addltErrCode) {}

template<typename errType>
std::string Anime4KCPP::ACException<errType, true>::what() const noexcept
{
    std::ostringstream source;
    source
        << "An error occurred. " << std::endl
        << std::endl
        << "Error type: " << errType::string << std::endl
        << std::endl
        << "Error message :" << std::endl
        << errorMessage << std::endl
        << std::endl
        << "Additional error code :" << std::endl
        << additionalErrorCode << std::endl
        << std::endl
        << "Additional information :" << std::endl
        << additionalInformation << std::endl;

    return source.str();
}
