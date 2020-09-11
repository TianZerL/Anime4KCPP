#include <jni.h>
#include <string>
#include "Anime4KCPP.h"

#ifdef DEBUG
#include <android/log.h>
#define TAG    "Anime4K-jni-test"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__)
#endif

extern "C"
{

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KCPU_createAnime4KCPU(
        JNIEnv *env,
        jobject /* this */) {
    return (jlong)(new Anime4KCPP::Anime4KCPU());
}

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

    return (jlong)(new Anime4KCPP::Anime4KCPU(
        Anime4KCPP::Parameters(
            passes,
            pushColorCount,
            strengthColor,
            strengthGradient,
            zoomFactor,
            fastMode,
            videoMode,
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
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_createAnime4KGPU(
        JNIEnv *env,
        jobject /* this */) {
    return (jlong)(new Anime4KCPP::Anime4KGPU());
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

    return (jlong)(new Anime4KCPP::Anime4KGPU(
        Anime4KCPP::Parameters(
            passes,
            pushColorCount,
            strengthColor,
            strengthGradient,
            zoomFactor,
            fastMode,
            videoMode,
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
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KCPUCNN_createAnime4KCPUCNN(
        JNIEnv *env,
        jobject /* this */) {
    return (jlong)(new Anime4KCPP::Anime4KCPUCNN());
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

    return (jlong)(new Anime4KCPP::Anime4KCPUCNN(
            Anime4KCPP::Parameters(
                    passes,
                    pushColorCount,
                    strengthColor,
                    strengthGradient,
                    zoomFactor,
                    fastMode,
                    videoMode,
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
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_createAnime4KGPUCNN(
        JNIEnv *env,
        jobject /* this */) {
    return (jlong)(new Anime4KCPP::Anime4KGPUCNN());
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

    return (jlong)(new Anime4KCPP::Anime4KGPUCNN(
            Anime4KCPP::Parameters(
                    passes,
                    pushColorCount,
                    strengthColor,
                    strengthGradient,
                    zoomFactor,
                    fastMode,
                    videoMode,
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
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_setArgumentsAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
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
        jbyte postFilters) {
    ((Anime4KCPP::Anime4K *)(ptrAnime4K))->setArguments(
        Anime4KCPP::Parameters(
            passes,
            pushColorCount,
            strengthColor,
            strengthGradient,
            zoomFactor,
            fastMode,
            videoMode,
            preprocessing,
            postprocessing,
            preFilters,
            postFilters
        )
    );
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_releaseAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
    delete ((Anime4KCPP::Anime4K*)(ptrAnime4K));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_loadImageAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring src) {
    try {
        ((Anime4KCPP::Anime4K*)(ptrAnime4K))->loadImage(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_processAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
    ((Anime4KCPP::Anime4K*)(ptrAnime4K))->process();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_processWithProgressAnime4K(
        JNIEnv *env,
        jobject thiz /* this */,
        jlong ptrAnime4K) {

    jclass classID = env->GetObjectClass(thiz);
    jmethodID callback = env->GetMethodID(classID,"progressCallback", "(DD)V");

    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();

    ((Anime4KCPP::Anime4K*)(ptrAnime4K))->processWithProgress(
            [&env, &thiz, &callback, &s](double v)
            {
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                env->CallVoidMethod(thiz, callback, v, std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0);
            });
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_saveImageAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring dst) {
    ((Anime4KCPP::Anime4K*)(ptrAnime4K))->saveImage(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_setVideoModeAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jboolean flag) {
    ((Anime4KCPP::Anime4K*)(ptrAnime4K))->setVideoMode(flag);
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_loadVideoAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring src) {
    try {
        ((Anime4KCPP::Anime4K*)(ptrAnime4K))->loadVideo(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_setVideoSaveInfoAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K,
        jstring dst) {
    try {
        ((Anime4KCPP::Anime4K*)(ptrAnime4K))->setVideoSaveInfo(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
    } catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4K_saveVideoAnime4K(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4K) {
    ((Anime4KCPP::Anime4K*)(ptrAnime4K))->saveVideo();
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_checkGPUSupportAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::Anime4KGPU::checkGPUSupport(0, 0).first);
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_initGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::Anime4KGPU::initGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_initGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::Anime4KGPUCNN::initGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_releaseGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::Anime4KGPU::releaseGPU();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_releaseGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    Anime4KCPP::Anime4KGPUCNN::releaseGPU();
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPU_isInitializedGPUAnime4KGPU(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::Anime4KGPU::isInitializedGPU());
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_wrapper_Anime4KGPUCNN_isInitializedGPUAnime4KGPUCNN(
        JNIEnv *env,
        jclass clazz) {
    return (jboolean)(Anime4KCPP::Anime4KGPUCNN::isInitializedGPU());
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
    std::pair<double, double> ret = Anime4KCPP::benchmark(0, 0);
    double retCppArray[] = {ret.first, ret.second};
    jdoubleArray retJavaArray = env->NewDoubleArray(2);
    env->SetDoubleArrayRegion(retJavaArray, 0, 2, retCppArray);
    return retJavaArray;
}

}
