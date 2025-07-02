#ifndef AC_UTIL_DEFER_HPP
#define AC_UTIL_DEFER_HPP

#include <utility>

namespace ac::util
{
    template <typename F>
    class Defer;
}

template <typename F>
class ac::util::Defer
{
public:
    explicit Defer(F&& f);
    Defer(const Defer<F>&) = delete;
    Defer(Defer<F>&&) = delete;
    Defer<F>& operator=(const Defer<F>&) = delete;
    Defer<F>& operator=(Defer<F>&&) = delete;
    ~Defer();
private:
    F func;
};

template<typename F>
inline ac::util::Defer<F>::Defer(F&& f) : func(std::forward<F>(f)) {}
template<typename F>
inline ac::util::Defer<F>::~Defer() { func(); }

#endif
