cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_2022_demo)

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
find_package(OpenCV CONFIG REQUIRED)

set(Torch_DIR "C:/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Add source to this project's executable.
add_executable(libtorch_2022_demo "libtorch_2022_demo.cpp" "libtorch_2022_demo.h")

# Specify the cuDNN paths
set(CUDNN_INCLUDE_DIR "C:/CuDNN/include")
set(CUDNN_LIBRARY "C:/CuDNN/lib/x64/cudnn.lib")

include_directories(${CUDNN_INCLUDE_DIR})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET libtorch_2022_demo PROPERTY CXX_STANDARD 20)
endif()

# Add include directories
target_include_directories(libtorch_2022_demo PRIVATE ${TORCH_INCLUDE_DIRS} "C:/libtorch/include")

# Link the Torch, OpenCV and cuDNN libraries
target_link_libraries(libtorch_2022_demo PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${CUDNN_LIBRARY})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET libtorch_2022_demo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:libtorch_2022_demo>)
endif (MSVC)