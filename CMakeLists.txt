cmake_minimum_required(VERSION 3.20)

project(flash_attention_v2 LANGUAGES C CXX CUDA)

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Find PyTorch
set(CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")
find_package(Torch REQUIRED)

set(CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC' -D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Source file
# Add target for simplePrintf
add_executable(flash_attention_v2 main.cpp flash_attention_v2.cu)

target_compile_options(flash_attention_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(flash_attention_v2 PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(flash_attention_v2 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH ON
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    cudart
    cublas
    cublasLt
    culibos
    pthread
    m
    dl
)