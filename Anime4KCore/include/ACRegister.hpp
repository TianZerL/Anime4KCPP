#undef REGISTER_PROCESSOR

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



//Register processor here
//---------------------------------------
#ifndef PROCESSORS
#define PROCESSORS \
REGISTER_PROCESSOR(CPU, Anime4K09) \
REGISTER_PROCESSOR(CPU, ACNet) \
REGISTER_PROCESSOR(OpenCL, Anime4K09) \
REGISTER_PROCESSOR(OpenCL, ACNet)
#endif
//---------------------------------------



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
