#ifndef __AC_REGISTER__
#define __AC_REGISTER__

#define REGISTER_PROCESSOR_IF(C, P, A) REGISTER_PROCESSOR_IF_IMPL(C, P, A) 
#define REGISTER_PROCESSOR_IF_IMPL(C, P, A) REGISTER_PROCESSOR_IF_##C(P, A) 

#define REGISTER_PROCESSOR_IF_1(P, A) REGISTER_PROCESSOR(P, A)
#define REGISTER_PROCESSOR_IF_0(P, A)

//Add flag here
//---------------------------------------
#ifdef ENABLE_OPENCL
#define OPENCL_FLAG 1
#else
#define OPENCL_FLAG 0
#endif

#ifdef ENABLE_CUDA
#define CUDA_FLAG 1
#else
#define CUDA_FLAG 0
#endif

#ifdef ENABLE_NCNN
#define NCNN_FLAG 1
#else
#define NCNN_FLAG 0
#endif
//---------------------------------------

//Register processor here
//---------------------------------------
#define PROCESSORS \
REGISTER_PROCESSOR(CPU, Anime4K09) \
REGISTER_PROCESSOR(CPU, ACNet) \
REGISTER_PROCESSOR_IF(OPENCL_FLAG ,OpenCL, Anime4K09) \
REGISTER_PROCESSOR_IF(OPENCL_FLAG ,OpenCL, ACNet) \
REGISTER_PROCESSOR_IF(CUDA_FLAG, Cuda, Anime4K09) \
REGISTER_PROCESSOR_IF(CUDA_FLAG, Cuda, ACNet) \
REGISTER_PROCESSOR_IF(NCNN_FLAG, NCNN, ACNet)
//---------------------------------------
#endif


#if defined(REGISTER_PROCESSOR)
#undef REGISTER_PROCESSOR
#endif

#if defined(AC_ENUM_ITEM)
#define REGISTER_PROCESSOR(P, A) P##_##A,
#endif

#if defined(AC_STREAM_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: stream << #P+std::string(" ")+#A; break;
#endif

#if defined(AC_CASE_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: return new P::A(parameters);
#endif

#if defined(AC_CASE_UP_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: return std::make_unique<P::A>(parameters);
#endif

#undef PROCESSOR_ENUM
#if defined(AC_ENUM_ITEM)
#define PROCESSOR_ENUM PROCESSORS
#endif


#undef PROCESSOR_STREAM
#if defined(AC_STREAM_ITEM)
#define PROCESSOR_STREAM PROCESSORS
#endif

#undef PROCESSOR_CASE
#if defined(AC_CASE_ITEM)
#define PROCESSOR_CASE PROCESSORS
#endif

#undef PROCESSOR_CASE_UP
#if defined(AC_CASE_UP_ITEM)
#define PROCESSOR_CASE_UP PROCESSORS
#endif
