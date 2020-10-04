cmake_minimum_required(VERSION 3.3)

project(test_Project)

file(GLOB SOURCES
    src/*.cpp)

add_executable(test_Project ${SOURCES})
include_directories(${CMAKE_SOURCE_DIR}/include)

if(WIN32)
    message( "Current generator: ${CMAKE_GENERATOR}" )
    set(CMAKE_GENERATOR "Visual Studio 15")
else()
    message( "Current generator: ${CMAKE_GENERATOR}" )
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4")
    set(CMAKE_GENERATOR "Unix Makefiles")
endif()

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    target_link_libraries(test_Project ${OpenCV_LIBS})
endif()




