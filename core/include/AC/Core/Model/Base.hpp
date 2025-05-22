#ifndef AC_CORE_MODEL_BASE_HPP
#define AC_CORE_MODEL_BASE_HPP

#define AC_CORE_SEQ_CNN_MODEL(model) \
public: \
    const float* kernels() const noexcept { return kptr; } \
    const float* biases() const noexcept { return bptr; } \
    static constexpr const char* name() noexcept { return #model; } \
private: \
    const float* kptr; \
    const float* bptr;

#endif
