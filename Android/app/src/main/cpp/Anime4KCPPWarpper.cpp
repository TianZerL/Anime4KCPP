#include <jni.h>
#include <string>
#include "Anime4K.h"
#include "Anime4KGPU.h"

extern "C"
{

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_createAnime4KCPP(
        JNIEnv *env,
        jobject /* this */) {
    return (jlong)(new Anime4K());
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_createAnime4KCPPByArgs(
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
        jbyte postFilters) {
    return (jlong)(new Anime4K(
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
            postFilters)
            );
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_setArgumentsAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
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
    ((Anime4K *)(ptrAnime4KCPP))->setArguments(
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
    );
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_releaseAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP) {
    delete ((Anime4K*)(ptrAnime4KCPP));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_loadImageAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
        jstring src) {
    try {
        ((Anime4K*)(ptrAnime4KCPP))->loadImage(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_processAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP) {
    ((Anime4K*)(ptrAnime4KCPP))->process();
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_saveImageAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
        jstring dst) {
    ((Anime4K*)(ptrAnime4KCPP))->saveImage(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_setVideoModeAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
        jboolean flag) {
    ((Anime4K*)(ptrAnime4KCPP))->setVideoMode(flag);
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_loadVideoAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
        jstring src) {
    try {
        ((Anime4K*)(ptrAnime4KCPP))->loadVideo(std::string(env->GetStringUTFChars(src, JNI_FALSE)));
    }
    catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_setVideoSaveInfoAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP,
        jstring dst) {
    try {
        ((Anime4K*)(ptrAnime4KCPP))->setVideoSaveInfo(std::string(env->GetStringUTFChars(dst, JNI_FALSE)));
    } catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_saveVideoAnime4KCPP(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP) {
        ((Anime4K*)(ptrAnime4KCPP))->saveVideo();
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPPGPU_createAnime4KCPPGPU(
        JNIEnv *env,
        jobject /* this */) {
    jlong ptrAnime4KCPPGPU = (jlong)nullptr;

    try {
        ptrAnime4KCPPGPU = (jlong)(new Anime4KGPU(true));
    } catch(const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }

    return ptrAnime4KCPPGPU;
}

JNIEXPORT jlong JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPPGPU_createAnime4KCPPGPUByArgs(
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
        jbyte postFilters) {
    jlong ptrAnime4KCPPGPU = (jlong)nullptr;

    try {
        ptrAnime4KCPPGPU = (jlong)(new Anime4KGPU(
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
            0,0, true)
        );
    } catch (const char* err) {
        env->ThrowNew(env->FindClass("java/lang/Exception"), err);
    }

    return ptrAnime4KCPPGPU;
}

JNIEXPORT void JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPPGPU_releaseAnime4KCPPGPU(
        JNIEnv *env,
        jobject /* this */,
        jlong ptrAnime4KCPP) {
    delete ((Anime4KGPU*)(ptrAnime4KCPP));
}

JNIEXPORT jboolean JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPPGPU_checkGPUSupportAnime4KCPPGPU(
        JNIEnv *env,
        jclass clazz) {
    return static_cast<jboolean>(Anime4KGPU::checkGPUSupport(0, 0).first);
}

JNIEXPORT jstring JNICALL
Java_github_tianzerl_anime4kcpp_Anime4KCPP_getCoreVersionAnime4KCPP(
        JNIEnv *env,
        jclass clazz) {
    return env->NewStringUTF(ANIME4KCPP_CORE_VERSION);
}

}
