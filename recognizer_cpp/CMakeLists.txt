cmake_minimum_required( VERSION 2.8 )
project( recognize_snooker )
find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS} ,
    "/usr/local/include/eigen3" ,
    "./includes/"
    "./src/headers/"
)

add_executable(
    recognize_snooker
    src/main.cpp
    src/Recognition.cpp
    src/Ball.cpp
)
target_link_libraries( recognize_snooker ${OpenCV_LIBS} )