#undef REGISTER_PROCESSOR

#if defined(AC_ENUM_ITEM)
#define REGISTER_PROCESSOR(P, A) P##_##A,
#elif defined(AC_STREAM_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: stream << #P+std::string(" ")+#A; break;
#elif defined(AC_CASE_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: return new P##::##A(parameters);
#elif defined(AC_CASE_SP_ITEM)
#define REGISTER_PROCESSOR(P, A) case Processor::Type::P##_##A: return std::make_shared<P##::##A>(parameters);
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



#if defined(AC_ENUM_ITEM)
#define PROCESSOR_ENUM PROCESSORS
#elif defined(AC_STREAM_ITEM)
#define PROCESSOR_STREAM PROCESSORS
#elif defined(AC_CASE_ITEM)
#define PROCESSOR_CASE PROCESSORS
#elif defined(AC_CASE_SP_ITEM)
#define PROCESSOR_CASE_SP PROCESSORS
#endif
