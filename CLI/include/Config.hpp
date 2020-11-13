#pragma once

#include <cmdline.h>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

std::string& ltrim(std::string& s)
{
    size_t pos = s.find_first_not_of(" \n\r\t");
    if (pos != std::string::npos)
        s.erase(0, pos);
    return s;
}

std::string& rtrim(std::string& s)
{
    size_t pos = s.find_last_not_of(" \n\r\t");
    if (pos != std::string::npos)
        s.erase(pos + 1);
    return s;
}

std::string& trim(std::string& s)
{
    ltrim(s);
    rtrim(s);
    return s;
}

struct ConfigError
{
    std::string filePath;
    std::string lineData;
    size_t lineNumber;
    size_t pos;

    void printErrorInfo()
    {
        std::cerr
            << "Invalid format: in file \"" << filePath << "\" line: " << lineNumber << ':' << pos << std::endl
            << lineData << std::endl
            << std::string(pos >= 1 ? pos : 0, ' ') << '^' << std::endl;
    }
};

class Config
{
public:
    Config(const cmdline::parser& parser = cmdline::parser())
        :parser(parser) {};

    cmdline::parser& getParser()
    {
        return parser;
    }

    Config& initFile(const std::string& path)
    {
        inputFile.open(path);
        err.filePath = path;
        return *this;
    }

    bool parseFile()
    {
        if (!inputFile.is_open())
        {
            setError("Unable to open config file", 0, 0);
            return false;
        }

        std::string currLine;

        size_t pos = 0;
        std::string key, value;
        size_t lineNumber = 0;
        std::string lineData;
        while (std::getline(inputFile, currLine))
        {
            lineNumber++;
            lineData = trim(currLine);

            if (currLine.empty())
                continue;

            if (currLine.front() == '#')
                continue;

            if ((pos = currLine.find('#')) != std::string::npos)
                rtrim(currLine.erase(pos));

            if ((pos = currLine.find('=')) != std::string::npos)
            {
                key = currLine.substr(0, pos);
                rtrim(key);
                if (key.empty())
                {
                    setError(lineData, lineNumber, 0);
                    return false;
                }

                value = currLine.substr(pos + 1);
                ltrim(value);
                if (value.empty())
                {
                    setError(lineData, lineNumber, pos + 1);
                    return false;
                }

                kvMap.emplace(key, value);
            }
            else
            {
                setError(lineData, lineNumber, pos);
                return false;
            }
        }
        return true;
    }

    template <typename T>
    T get(const std::string& key)
    {
        if (!inputFile.is_open())
        {
            const T& tmp = parser.get<T>(key);
            kvMap.emplace(key, std::to_string(tmp));
            return tmp;
        }
        auto it = kvMap.find(key);
        if (it == kvMap.end())
        {
            const T& tmp = parser.get<T>(key);
            kvMap.emplace(key, std::to_string(tmp));
            return tmp;
        }
        std::istringstream converter(it->second);
        T ret{};
        converter >> ret;
        return ret;
    }

    template <>
    std::string get(const std::string& key)
    {
        if (!inputFile.is_open())
        {
            const std::string& tmp = parser.get<std::string>(key);
            kvMap.emplace(key, tmp);
            return tmp;
        }
        auto it = kvMap.find(key);
        if (it == kvMap.end())
        {
            const std::string& tmp = parser.get<std::string>(key);
            kvMap.emplace(key, tmp);
            return tmp;
        }
        return it->second;
    }

    template <>
    bool get<bool>(const std::string& key)
    {
        if (!inputFile.is_open())
        {
            bool tmp = parser.exist(key);
            kvMap.emplace(key, std::to_string(tmp));
            return tmp;
        }
        auto it = kvMap.find(key);
        if (it == kvMap.end())
        {
            bool tmp = parser.exist(key);
            kvMap.emplace(key, std::to_string(tmp));
            return tmp;
        }
        std::istringstream converter(it->second);
        bool ret = false;
        converter >> ret;
        return ret;
    }

    std::unordered_map<std::string, std::string>& getMap()
    {
        return kvMap;
    }

    ConfigError getLastError()
    {
        return err;
    }

private:
    void setError(const std::string& lineData, size_t lineNumber, size_t pos)
    {
        err.lineData = lineData;
        err.lineNumber = lineNumber;
        err.pos = pos;
    }

private:
    std::unordered_map<std::string, std::string> kvMap;
    std::ifstream inputFile;

    ConfigError err;
    cmdline::parser parser;
};

class ConfigWriter
{
public:
    ConfigWriter() = default;
    bool initFile(const std::string& path)
    {
        outputFile.open(path);
        return outputFile.is_open();
    }

    void write()
    {
        if (!outputFile.is_open())
            return;
        outputFile <<"# Anime4KCPP_CLI config file\n" << std::endl;
        for (auto& kv : kvMap)
        {
            outputFile << kv.first << " = " << kv.second << std::endl;
        }
    }

    void set(Config& config)
    {
        kvMap = config.getMap();
    }

private:
    std::unordered_map<std::string, std::string> kvMap;
    std::ofstream outputFile;
};
