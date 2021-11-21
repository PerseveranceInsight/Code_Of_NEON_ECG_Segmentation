# export EN_DEBUG_SYM=true

# export EN_MATRIX_DBG=true
# export EN_MODEL_DBG=true
# export EN_PACK_DEBUG=true
# export EN_GEMM_DBG=true
export EN_GRAPH_DBG=true
# export EN_SIGNAL_DBG=true
# export EN_SIG2COL_DBG=true

ndk-build -B APP_BUILD_SCRIPT=Android.mk NDK_PROJECT_PATH=. APP_ABI=arm64-v8a NDK_APPLICATION_MK=Application.mk NDK_DEBUG=1
