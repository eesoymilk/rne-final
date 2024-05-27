cmake_minimum_required(VERSION 3.4)

project(DataServer)

find_package(Threads)

file(GLOB SOURCE src/*.cpp)
file(GLOB HEADER inc/*.h)



#add_library(DataServer ${SOURCE} ${HEADER})
add_executable(DataServer ${SOURCE} ${HEADER})
target_link_libraries(DataServer ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(DataServer PUBLIC inc)
target_include_directories(DataServer PUBLIC ext/websocketpp/include)
target_include_directories(DataServer PUBLIC ext/asio-1.12.1/include)
add_definitions("-DASIO_STANDALONE")
add_definitions("-D_WEBSOCKETPP_CPP11_TYPE_TRAITS_")
