#include <jni.h>
#include <string>
#include "Anime4KCPP.hpp"

#ifdef DEBUG
#include <android/log.h>
#define TAG    "Anime4K-jni-test"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__)
#endif

extern "C"
{

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KCPU_createAnime4KCPUByArgs(
        JNIEnv *env,
        jobject /* this */,
        jint passes,
        jint pushColorCount,
        jdouble strengthColor,
        jdouble strengthGradient,
        jdouble zoomFactor,
        jboolean fastMode,
        jboolean videoMode,
        jboolean preprocessing,
        jboolean postprocessing,
        jbyte preFilters,
        jbyte postFilters,
        jboolean alpha) {

    return (jlong)(new Anime4KCPP::CPU::Anime4K09(
        Anime4KCPP::Parameters(
            passes,
            pushColorCount,
            strengthColor,
            strengthGradient,
            zoomFactor,
            fastMode,
            preprocessing,
            postprocessing,
            preFilters,
            postFilters,
            std::thread::hardware_concurrency(),
            false,
            1,
            alpha
            ))
    );
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_createAnime4KGPUByArgs(
        JNIEnv *env,
        jobject /* this */,
        jint passes,
        jint pushColorCount,
        jdouble strengthColor,
        jdouble strengthGradient,
        jdouble zoomFactor,
        jboolean fastMode,
        jboolean videoMode,
        jboolean preprocessing,
        jboolean postprocessing,
        jbyte preFilters,
        jbyte postFilters,
        jboolean alpha) {

    return (jlong)(new Anime4KCPP::OpenCL::Anime4K09(
        Anime4KCPP::Parameters(
            passes,
            pushColorCount,
            strengthColor,
            strengthGradient,
            zoomFactor,
            fastMode,
            preprocessing,
            postprocessing,
            preFilters,
            postFilters,
            std::thread::hardware_concurrency(),
            false,
            1,
            alpha
        ))
    );
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KCPUCNN_createAnime4KCPUCNNByArgs(
        JNIEnv *env,
        jobject /* this */,
        jint passes,
        jint pushColorCount,
        jdouble strengthColor,
        jdouble strengthGradient,
        jdouble zoomFactor,
        jboolean fastMode,
        jboolean videoMode,
        jboolean preprocessing,
        jboolean postprocessing,
        jbyte preFilters,
        jbyte postFilters,
        jboolean HDN,
        jint HDNLevel,
        jboolean alpha) {

    return (jlong)(new Anime4KCPP::CPU::ACNet(
            Anime4KCPP::Parameters(
                    passes,
                    pushColorCount,
                    strengthColor,
                    strengthGradient,
                    zoomFactor,
                    fastMode,
                    preprocessing,
                    postprocessing,
                    preFilters,
                    postFilters,
                    std::thread::hardware_concurrency(),
                    HDN,
                    HDNLevel,
                    alpha
            ))
    );
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_createAnime4KGPUCNNByArgs(
        JNIEnv *env,
        jobject /* this */,
        jint passes,
        jint pushColorCount,
        jdouble strengthColor,
        jdouble strengthGradient,
        jdouble zoomFactor,
        jboolean fastMode,
        jboolean videoMode,
        jboolean preprocessing,
        jboolean postprocessing,
        jbyte preFilters,
        jbyte postFilters,
        jboolean HDN,
        jint HDNLevel,
        jboolean alpha) {

    return (jlong)(new Anime4KCPP::OpenCL::ACNet(
            Anime4KCPP::Parameters(
                    passes,
                    pushColorCount,
                    strengthColor,
                    strengthGradient,
                    zoomFactor,
                    fastMode,
                    preprocessing,
                    postprocessing,
                    preFilters,
                    postFilters,
                    std::thread::hardware_concurrency(),
                    HDN,
                    HDNLevel,
                    alpha
            ))
    );
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_releaseAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
    delete ((Anime4KCPP::AC*)(ptrAnime4K));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_loadImageAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring src) {
    try {
        ((Anime4KCPP::AC*)(ptrAnime4K))->loadImage(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const std::exception& err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err.what());
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_processAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
    ((Anime4KCPP::AC*)(ptrAnime4K))->process();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_saveImageAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring dst) {
    ((Anime4KCPP::AC*)(ptrAnime4K))->saveImage(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_createVideoProcessor(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
        return (jlong)(new Anime4KCPP::VideoProcessor(*((Anime4KCPP::AC*)ptrAnime4K)));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_releaseVideoProcessor(
        JNIEnv *env,
        jobject /* this */,
        jlong ptr) {
    delete ((Anime4KCPP::VideoProcessor*)(ptr));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_loadVideoVideoProcessor(
        JNIEnv *env,
        jobject /* this */,
        jlong ptr,
        jstring src) {
    try {
        ((Anime4KCPP::VideoProcessor*)(ptr))->loadVideo(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const std::exception& err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err.what());
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_setVideoSaveInfoVideoProcessor(
        JNIEnv *env,
        jobject /* this */,
        jlong ptr,
        jstring dst) {
    try {
        ((Anime4KCPP::VideoProcessor*)(ptr))->setVideoSaveInfo(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
    }
    catch (const std::exception& err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err.what());
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_saveVideoVideoProcessor(
        JNIEnv *env,
        jobject /* this */,
        jlong ptr) {
    ((Anime4KCPP::VideoProcessor*)(ptr))->saveVideo();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_VideoProcessor_processWithProgressVideoProcessor(
        JNIEnv *env,
        jobject thiz /* this */,
        jlong ptr) {

    jclass classID = env->GetObjectClass(thiz);
    jmethodID callback = env->GetMethodID(classID,"progressCallback", "(DD)V");

    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();

    ((Anime4KCPP::VideoProcessor*)(ptr))->processWithProgress(
            [&env, &thiz, &callback, &s](double v)
            {
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                env->CallVoidMethod(thiz, callback, v, std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
            });
}


JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_checkGPUSupportAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::OpenCL::checkGPUSupport(0, 0).supported);
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_initGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::OpenCL::Anime4K09::initGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_initGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::OpenCL::ACNet::initGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_releaseGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::OpenCL::Anime4K09::releaseGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_releaseGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::OpenCL::ACNet::releaseGPU();
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_isInitializedGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU());
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_isInitializedGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::OpenCL::ACNet::isInitializedGPU());
}


JNIEXPORT jstring JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_getCoreVersionAnime4K(
        JNIEnv *env,
        jclass clazz) {
    return env->NewStringUTF(ANIME4KCPP_CORE_VERSION);
}

JNIEXPORT jdoubleArray JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_benchmarkAnime4K(
        JNIEnv *env,
        jclass clazz) {
    double CPUScore = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 720, 480>();
    double GPUScore = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 720, 480>(0, 0);
    double retCppArray[] = {CPUScore, GPUScore};
    jdoubleArray retJavaArray = env->NewDoubleArray(2);
    env->SetDoubleArrayRegion(retJavaArray, 0, 2, retCppArray);
    return retJavaArray;
}

}
