
message( "Starting Script - Current generator: ${CMAKE_GENERATOR}" )

project(test_Project)
cmake_minimum_required(VERSION 3.3)

if(WIN32)
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/win")
else()
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/osx/lib/cmake/opencv4")
endif()


file(GLOB SOURCES
    src/*.cpp)

add_executable(test_Project ${SOURCES})
include_directories(${CMAKE_SOURCE_DIR}/include)


find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    target_link_libraries(test_Project ${OpenCV_LIBS})
endif()




