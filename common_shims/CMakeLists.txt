cmake_minimum_required(VERSION 3.14)
project(llama-cpp-shims LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp-master")

find_package(nlohmann_json 3.11.3 QUIET CONFIG)
if(NOT nlohmann_json_FOUND)
    # Option 2: Fallback to FetchContent
    message(STATUS "nlohmann_json not found → using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
else()
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
endif()

add_library(common-shims STATIC common_shims.cpp)

target_include_directories(common-shims PUBLIC
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/common
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(common-shims PUBLIC
    nlohmann_json::nlohmann_json
)

set_target_properties(common-shims PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/llama.cpp-build/common"
    OUTPUT_NAME "common-shims"
)
