file(GLOB MACE_RUN_SRCS
        ssd-mobilenet.cpp
        )
SET(OpenCV_DIR /opt/3rdparty/opencv-3.2.0/build)
FIND_PACKAGE(OpenCV REQUIRED)


add_executable(ssd-mobilenet ${MACE_RUN_SRCS})
target_link_libraries(ssd-mobilenet PUBLIC
        mace_static
        model
        gflags
        ${OpenCV_LIBS}
        )
file(GLOB MACE_RUN_SRCS
        openpose-mobilenet.cpp pafprocess.cpp Human.cpp
        )
add_executable(openpose-mobilenet ${MACE_RUN_SRCS})
target_link_libraries(openpose-mobilenet PUBLIC
        mace_static
        model
        gflags
        ${OpenCV_LIBS}
        )

install(TARGETS ssd-mobilenet RUNTIME DESTINATION bin)
