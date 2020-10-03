cmake_minimum_required(VERSION 3.3)

if(WIN32)
    set(CMAKE_GENERATOR "Visual Studio 15")
else
    set(CMAKE_GENERATOR "Unix Makefiles")
endif()



add_executable(intro main.cpp)



