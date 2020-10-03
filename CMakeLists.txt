cmake_minimum_required(VERSION 3.3)

if(WIN32)
    message( "Current generator: ${CMAKE_GENERATOR}" )
    set(CMAKE_GENERATOR "Visual Studio 15")
else
    message( "Current generator: ${CMAKE_GENERATOR}" )
    set(CMAKE_GENERATOR "Unix Makefiles")
endif()



add_executable(intro main.cpp)



