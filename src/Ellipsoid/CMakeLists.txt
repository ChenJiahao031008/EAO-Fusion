# set(CMAKE_BUILD_TYPE Debug)

add_library(Ellipsoid SHARED
    Ellipsoid.cpp
)

target_link_libraries(Ellipsoid
        ${OpenCV_LIBS}
        symmetry
        utils
        )
