cmake_minimum_required(VERSION 3.6)
project(dR_c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c11")

include_directories(inc)
#include_directories(model_weight)
include_directories(model_weight_BK_20170308)

set(SOURCE_FILES
        src/main.c
        src/recognizer.c
        src/imageIO.c
        src/mat.c
        src/cnn.c
        )

add_executable(dR_c ${SOURCE_FILES})
# linux 下需要手动添加 math.h ， 即 make *** ***.o -lm,
# reference : http://stackoverflow.com/questions/15181899/cmake-correctly-linking-system-library-using-gcc
# cmake 下设置如下：
target_link_libraries(dR_c m)
