cmake_minimum_required (VERSION 2.8.11)

project(PitchSpeed)

# GENERAL DEFINITIONS
add_definitions(-D_FILE_OFSET_BITS=64)
add_definitions(-DDATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}bin/\")
add_definitions(-I /usr/include)
add_definitions(-I /usr/local/include)

# BUG: (12-Jan-2015) CHANGES FOR VISUAL STUDIO PROJECT CONFIGURATION
add_definitions(-DNOMINMAX)

if ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
add_definitions(-fpermissive -std=c++11)
endif()

# BUG: (12-Jan-2015) CHANGES FOR VISUAL STUDIO PROJECT CONFIGURATION
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
# BUG: (12-Jan-2015) CHANGES FOR VISUAL STUDIO PROJECT CONFIGURATION
add_definitions(-D_SCL_SECURE_NO_DEPRECATE)

include_directories(source)

# BOOST
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.59 COMPONENTS system filesystem timer date_time REQUIRED)

LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

# CURL
find_package(CURL REQUIRED)

# RAPIDJSON
find_package(RAPIDJSON)

# PUGIXML
# https://github.com/imageworks/OpenShadingLanguage/blob/master/src/cmake/modules/FindPugiXML.cmake
# Find the pugixml XML parsing library.
#
# Sets the usual variables expected for find_package scripts:
#
# PUGIXML_INCLUDE_DIR - header location
# PUGIXML_LIBRARIES - library to link against
# PUGIXML_FOUND - true if pugixml was found.


add_subdirectory(source/SpeedFrom2dData)

