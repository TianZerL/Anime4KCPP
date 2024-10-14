#include <cassert>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#if __has_include(<CL/opencl.hpp>)
#   include <CL/opencl.hpp>
#elif __has_include(<CL/cl2.hpp>)
#   include <CL/cl2.hpp>
#else
#   include <CL/cl.hpp>
#endif

#include "AC/Core/Processor.hpp"
#include "AC/Core/Util.hpp"
#include "AC/Core/Model/ACNet.hpp"
#include "AC/Core/OpenCL/Kernel.hpp" // Generated by CMake
#include "AC/Util/ThreadLocal.hpp"

#include "ACExport.hpp" // Generated by CMake

namespace ac::core::opencl
{
    struct Context
    {
        std::string name;
        cl::Device device;
        cl::Context ctx;
        cl::Program program;
    };

    // we cannot make ContextList as static like cuda.
    // it will crash while unload the DLL(if build this into a DLL), god knows why.
    inline static std::vector<Context> getContextList() noexcept
    {
        std::vector<Context> contexts{};
        std::vector<cl::Platform> platforms{};
        cl::Platform::get(&platforms);
        for (auto&& platform : platforms)
        {
            std::vector<cl::Device> devices{};
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto&& device : devices)
            {
                std::string name{};
                device.getInfo(CL_DEVICE_NAME, &name);
                contexts.emplace_back(Context{ name, device, {}, {} });
            }
        }
        return contexts;
    }

    // we can call `init` multiple times
    inline static cl_int init(Context& context, const char* const kernel) noexcept
    {
        if (!context.device()) return CL_DEVICE_NOT_AVAILABLE;
        if (context.ctx() && context.program()) return CL_SUCCESS;

        cl_int err = CL_SUCCESS;
        context.ctx = cl::Context{ context.device, nullptr, nullptr, nullptr, &err }; if (err != CL_SUCCESS) return err;
        context.program = cl::Program{ context.ctx, kernel, false, &err }; if (err != CL_SUCCESS) return err;
        return context.program.build(context.device);
    }
    inline static cl_channel_type channelType(const Image::ElementType elementType) noexcept
    {
        switch (elementType)
        {
        case Image::UInt8: return CL_UNORM_INT8;
        case Image::UInt16: return CL_UNORM_INT16;
        case Image::Float32: return CL_FLOAT;
        default: return assert(elementType == Image::UInt8 || elementType == Image::UInt16 || elementType == Image::Float32), 0;
        }
    }

    class OpenCLProcessorBase : public Processor
    {
    public:
        OpenCLProcessorBase(const int device) noexcept
        {
            auto& err = errors.local();
            auto contextList = getContextList();
            if (contextList.empty()) err = CL_DEVICE_NOT_FOUND;
            else
            {
                idx = (device >= 0 && static_cast<decltype(contextList.size())>(device) < contextList.size()) ? device : 0;
                context = contextList[idx];
                err = init(context, KernelString/*from Kernel.hpp*/);
            }
        }
        ~OpenCLProcessorBase() noexcept override = default;

        bool ok() noexcept override
        {
            return errors.local() == CL_SUCCESS;
        }
        const char* error() noexcept override
        {
            switch (errors.local())
            {// run-time and JIT compiler errors
            case 0: return "CL_SUCCESS";
            case -1: return "CL_DEVICE_NOT_FOUND";
            case -2: return "CL_DEVICE_NOT_AVAILABLE";
            case -3: return "CL_COMPILER_NOT_AVAILABLE";
            case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5: return "CL_OUT_OF_RESOURCES";
            case -6: return "CL_OUT_OF_HOST_MEMORY";
            case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8: return "CL_MEM_COPY_OVERLAP";
            case -9: return "CL_IMAGE_FORMAT_MISMATCH";
            case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11: return "CL_BUILD_PROGRAM_FAILURE";
            case -12: return "CL_MAP_FAILURE";
            case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15: return "CL_COMPILE_PROGRAM_FAILURE";
            case -16: return "CL_LINKER_NOT_AVAILABLE";
            case -17: return "CL_LINK_PROGRAM_FAILURE";
            case -18: return "CL_DEVICE_PARTITION_FAILED";
            case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            // compile-time errors
            case -30: return "CL_INVALID_VALUE";
            case -31: return "CL_INVALID_DEVICE_TYPE";
            case -32: return "CL_INVALID_PLATFORM";
            case -33: return "CL_INVALID_DEVICE";
            case -34: return "CL_INVALID_CONTEXT";
            case -35: return "CL_INVALID_QUEUE_PROPERTIES";
            case -36: return "CL_INVALID_COMMAND_QUEUE";
            case -37: return "CL_INVALID_HOST_PTR";
            case -38: return "CL_INVALID_MEM_OBJECT";
            case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40: return "CL_INVALID_IMAGE_SIZE";
            case -41: return "CL_INVALID_SAMPLER";
            case -42: return "CL_INVALID_BINARY";
            case -43: return "CL_INVALID_BUILD_OPTIONS";
            case -44: return "CL_INVALID_PROGRAM";
            case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46: return "CL_INVALID_KERNEL_NAME";
            case -47: return "CL_INVALID_KERNEL_DEFINITION";
            case -48: return "CL_INVALID_KERNEL";
            case -49: return "CL_INVALID_ARG_INDEX";
            case -50: return "CL_INVALID_ARG_VALUE";
            case -51: return "CL_INVALID_ARG_SIZE";
            case -52: return "CL_INVALID_KERNEL_ARGS";
            case -53: return "CL_INVALID_WORK_DIMENSION";
            case -54: return "CL_INVALID_WORK_GROUP_SIZE";
            case -55: return "CL_INVALID_WORK_ITEM_SIZE";
            case -56: return "CL_INVALID_GLOBAL_OFFSET";
            case -57: return "CL_INVALID_EVENT_WAIT_LIST";
            case -58: return "CL_INVALID_EVENT";
            case -59: return "CL_INVALID_OPERATION";
            case -60: return "CL_INVALID_GL_OBJECT";
            case -61: return "CL_INVALID_BUFFER_SIZE";
            case -62: return "CL_INVALID_MIP_LEVEL";
            case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64: return "CL_INVALID_PROPERTY";
            case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
            case -66: return "CL_INVALID_COMPILER_OPTIONS";
            case -67: return "CL_INVALID_LINKER_OPTIONS";
            case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
            case -69: return "CL_INVALID_PIPE_SIZE";
            case -70: return "CL_INVALID_DEVICE_QUEUE";
            case -71: return "CL_INVALID_SPEC_ID";
            case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
            // extension errors
            case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
            default: return "CL_UNKNOWN_ERROR";
            }
        }
        const char* name() const noexcept override
        {
            return context.name.c_str();
        }
    protected:
        cl::CommandQueue& queue(cl_int* const err)
        {
            auto& cmdq = queues.local();
            if (!cmdq()) cmdq = cl::CommandQueue{ context.ctx, context.device, 0, err };
            return cmdq;
        }
    protected:
        Context context;
        util::ThreadLocal<cl_int> errors;
        util::ThreadLocal<cl::CommandQueue> queues;
    };

    template<typename Model>
    class OpenCLProcessor;
}

template<>
class ac::core::opencl::OpenCLProcessor<ac::core::model::ACNet> : public ac::core::opencl::OpenCLProcessorBase
{
public:
    OpenCLProcessor(int device, const model::ACNet& model) noexcept;
    ~OpenCLProcessor() noexcept override;
private:
    void process(const Image& src, Image& dst) override;
private:
    cl::Buffer kernels;
    cl::Buffer biases;
};

ac::core::opencl::OpenCLProcessor<ac::core::model::ACNet>::OpenCLProcessor(const int device, const model::ACNet& model) noexcept : OpenCLProcessorBase(device)
{
    auto& err = errors.local();
    if (err != CL_SUCCESS) return; // check if initialization was successful
    kernels = cl::Buffer{context.ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, model.kernelSize(), const_cast<float*>(model.kernels()), &err}; if (err != CL_SUCCESS) return;
    biases = cl::Buffer{context.ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, model.biasSize(), const_cast<float*>(model.biases()), &err};
}
ac::core::opencl::OpenCLProcessor<ac::core::model::ACNet>::~OpenCLProcessor() noexcept = default;

void ac::core::opencl::OpenCLProcessor<ac::core::model::ACNet>::process(const Image& src, Image& dst)
{
    cl::size_type srcW = src.width(), srcH = src.height();
    cl::size_type dstW = dst.width(), dstH = dst.height();
    cl::size_type srcRangeW = align(srcW, 16), srcRangeH = align(srcH, 8);
    cl::size_type dstRangeW = align(dstW, 16), dstRangeH = align(dstH, 8);

    auto& err = errors.local();
    auto& cmdq = queue(&err); if (err != CL_SUCCESS) return;

    cl::Kernel conv3x3_1to8{ context.program, "conv3x3_1to8", &err }; if (err != CL_SUCCESS) return;
    cl::Kernel conv3x3_8to8{ context.program, "conv3x3_8to8", &err }; if (err != CL_SUCCESS) return;
    cl::Kernel deconv2x2_8to1{ context.program, "deconv2x2_8to1", &err }; if (err != CL_SUCCESS) return;

    cl::Image2D in{ context.ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, {CL_R, channelType(src.type())}, srcW, srcH, 0, nullptr, &err }; if (err != CL_SUCCESS) return;
    cl::Image2DArray tmp1{ context.ctx, CL_MEM_READ_WRITE, {CL_RGBA, CL_HALF_FLOAT}, 2, srcW, srcH, 0, 0, nullptr, &err }; if (err != CL_SUCCESS) return;
    cl::Image2DArray tmp2{ context.ctx, CL_MEM_READ_WRITE, {CL_RGBA, CL_HALF_FLOAT}, 2, srcW, srcH, 0, 0, nullptr, &err }; if (err != CL_SUCCESS) return;
    cl::Image2D out{ context.ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, {CL_R, channelType(dst.type())}, dstW, dstH, 0, nullptr, &err }; if (err != CL_SUCCESS) return;

    err = cmdq.enqueueWriteImage(in, CL_FALSE, { 0,0,0 }, { srcW,srcH,1 }, src.stride(), 0, src.ptr()); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(0, in); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(1, tmp1); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(2, kernels); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(3, model::ACNet::kernelOffset[0]); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(4, biases); if (err != CL_SUCCESS) return;
    err = conv3x3_1to8.setArg(5, model::ACNet::baisOffset[0]); if (err != CL_SUCCESS) return;
    err = cmdq.enqueueNDRangeKernel(conv3x3_1to8, cl::NullRange, { srcRangeW, srcRangeH }, { 16, 8 }); if (err != CL_SUCCESS) return;
    for (int i = 0; i < 4; i++)
    {
        err = conv3x3_8to8.setArg(0, tmp1); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(1, tmp2); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(2, kernels); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(3, model::ACNet::kernelOffset[i * 2 + 1]); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(4, biases); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(5, model::ACNet::baisOffset[i * 2 + 1]); if (err != CL_SUCCESS) return;
        err = cmdq.enqueueNDRangeKernel(conv3x3_8to8, cl::NullRange, { srcRangeW, srcRangeH }, { 16, 8 }); if (err != CL_SUCCESS) return;

        err = conv3x3_8to8.setArg(0, tmp2); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(1, tmp1); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(2, kernels); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(3, model::ACNet::kernelOffset[i * 2 + 2]); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(4, biases); if (err != CL_SUCCESS) return;
        err = conv3x3_8to8.setArg(5, model::ACNet::baisOffset[i * 2 + 2]); if (err != CL_SUCCESS) return;
        err = cmdq.enqueueNDRangeKernel(conv3x3_8to8, cl::NullRange, { srcRangeW, srcRangeH }, { 16, 8 }); if (err != CL_SUCCESS) return;
    }
    err = deconv2x2_8to1.setArg(0, tmp1); if (err != CL_SUCCESS) return;
    err = deconv2x2_8to1.setArg(1, out); if (err != CL_SUCCESS) return;
    err = deconv2x2_8to1.setArg(2, kernels); if (err != CL_SUCCESS) return;
    err = deconv2x2_8to1.setArg(3, model::ACNet::kernelOffset[9]); if (err != CL_SUCCESS) return;
    err = cmdq.enqueueNDRangeKernel(deconv2x2_8to1, cl::NullRange, { dstRangeW, dstRangeH }, { 16, 8 }); if (err != CL_SUCCESS) return;
    err = cmdq.enqueueReadImage(out, CL_TRUE, { 0,0,0 }, { dstW,dstH,1 }, dst.stride(), 0, dst.ptr());
}

template<>
AC_EXPORT std::shared_ptr<ac::core::Processor> ac::core::Processor::create<ac::core::Processor::OpenCL, ac::core::model::ACNet>(const int idx, const model::ACNet& model)
{
    return std::make_shared<opencl::OpenCLProcessor<model::ACNet>>(idx, model);
}
template<>
AC_EXPORT const char* ac::core::Processor::info<ac::core::Processor::OpenCL>()
{
    static auto infoBuffer = []() -> std::string {
        auto contextList = opencl::getContextList();
        std::ostringstream buffer{ "OpenCL:\n", std::ios_base::ate };
        for (std::size_t i = 0; i < contextList.size(); i++)
            buffer << "  [" << i << "] " << contextList[i].name << '\n';
        return buffer.str();
    }();
    return infoBuffer.c_str();
}
