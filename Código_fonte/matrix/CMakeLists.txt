cmake_minimum_required(VERSION 2.8)
project( CriaImagem )
include(FetchContent)
      FetchContent_Declare(dlib
      GIT_REPOSITORY https://github.com/davisking/dlib.git
      GIT_TAG        v19.18)
FetchContent_MakeAvailable(dlib)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( CriaImagem CriaImagem.cpp )
target_link_libraries(CriaImagem dlib::dlib ${OpenCV_LIBS} )
