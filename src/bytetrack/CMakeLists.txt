project (bytetrack)

file(GLOB _SRCS "*.cc" "*.h" "*.hpp")

add_library(${PROJECT_NAME} SHARED ${_SRCS})
include_directories(${PROJECT_NAME} PUBLIC
    /usr/local/include/eigen3
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    nvinfer
    cudart
    ${OpenCV_LIBS}
)
