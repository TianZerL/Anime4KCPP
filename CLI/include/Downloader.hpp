#ifdef ENABLE_LIBCURL

#include <string>
#include <vector>

class Downloader
{
public:
    ~Downloader();

    void init();
    void release() noexcept;
    void download(std::string url, const std::vector<unsigned char>& buf);

private:
    void* curl = nullptr;

};

#endif // ENABLE_LIBCURL
