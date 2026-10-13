cmake_minimum_required(VERSION 3.15)

project(DecanCore LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(DECAN_ENABLE_NEOML "Build and link the bundled NeoML library" ON)
option(DECAN_ENABLE_ONNXRUNTIME "Link an installed ONNX Runtime target when available" OFF)
option(DECAN_SETUP_DUMMY_MODELS
    "Provision deterministic test ONNX models as a build dependency" ON)
set(DECAN_MODELS_DIR "${CMAKE_BINARY_DIR}/models" CACHE PATH
    "Directory containing the four runtime ONNX models")

# Compatibility aliases for existing build scripts.
if(DEFINED ENABLE_NEOML)
    set(DECAN_ENABLE_NEOML "${ENABLE_NEOML}" CACHE BOOL
        "Build and link the bundled NeoML library" FORCE)
endif()
if(DEFINED MODELS_DIR)
    set(DECAN_MODELS_DIR "${MODELS_DIR}" CACHE PATH
        "Directory containing the four runtime ONNX models" FORCE)
endif()


# Находим OpenCV

find_package(OpenCV REQUIRED)
if(DECAN_SETUP_DUMMY_MODELS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(DECAN_ONNXRUNTIME_TARGET "")
if(DECAN_ENABLE_ONNXRUNTIME)
    # NeoML uses its bundled NeoOnnx importer by default. ONNX Runtime is an
    # optional target for deployments that expose it from a CMake package; the
    # backend does not require it when NeoML/NeoOnnx is enabled.
    find_package(onnxruntime CONFIG QUIET)
    if(TARGET onnxruntime::onnxruntime)
        set(DECAN_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
    elseif(TARGET onnxruntime)
        set(DECAN_ONNXRUNTIME_TARGET onnxruntime)
    endif()
endif()

if(DECAN_ENABLE_NEOML AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/neoml/NeoML/CMakeLists.txt")
    # Use the protobuf installed by Homebrew instead of downloading another copy.
    set(NEOML_USE_SYSTEM_PROTOBUF ON CACHE BOOL "Use system protobuf for NeoML")
    set(NeoML_BUILD_TESTS OFF CACHE BOOL "Disable NeoML tests in the application build")
    set(NeoML_UNITY_BUILD ON CACHE BOOL "Use unity build for NeoML")
    set(NeoML_BUILD_SHARED OFF CACHE BOOL "Build NeoML as a static library")
    set(NeoOnnx_BUILD_SHARED OFF CACHE BOOL "Build NeoOnnx as a static library")
    set(NeoOnnx_BUILD ON CACHE BOOL "Build NeoOnnx ONNX importer")
    set(NeoMathEngine_ENABLE_VULKAN OFF CACHE BOOL "Disable Vulkan backend")
    set(NeoML_INSTALL OFF CACHE BOOL "Do not install NeoML during application build")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/neoml/NeoML
                     ${CMAKE_CURRENT_BINARY_DIR}/neoml EXCLUDE_FROM_ALL)
endif()

set(DECAN_NEOML_LINKED OFF)
if(DECAN_ENABLE_NEOML AND TARGET NeoML AND TARGET NeoOnnx)
    set(DECAN_NEOML_LINKED ON)
    message(STATUS "Decan NeoML backend: ENABLED (NeoML + NeoOnnx)")
elseif(DECAN_ENABLE_NEOML)
    message(WARNING
        "DECAN_ENABLE_NEOML=ON, but NeoML and NeoOnnx targets are unavailable; "
        "the application will use the fallback backend")
else()
    message(STATUS "Decan NeoML backend: DISABLED (fallback build)")
endif()


add_library(decan_document_processor
    src/document_processor.cpp
    src/neoml_onnx_backend.cpp
    src/text_post_processor.cpp
)
target_include_directories(decan_document_processor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_features(decan_document_processor PUBLIC cxx_std_17)
target_link_libraries(decan_document_processor PUBLIC ${OpenCV_LIBS})
if(DECAN_NEOML_LINKED)
    target_link_libraries(decan_document_processor PUBLIC NeoML NeoOnnx)
    # DECAN_WITH_NEOML is the source-level switch. The aliases are exported as
    # well so integrations can use the same CMake build with older adapter code.
    target_compile_definitions(decan_document_processor PUBLIC
        DECAN_WITH_NEOML=1
        USE_NEOML=1
        ENABLE_NEOML=1
    )
endif()
if(DECAN_ONNXRUNTIME_TARGET)
    target_link_libraries(decan_document_processor PUBLIC ${DECAN_ONNXRUNTIME_TARGET})
    target_compile_definitions(decan_document_processor PUBLIC DECAN_WITH_ONNXRUNTIME=1)
endif()


add_executable(decan_core main.cpp)

target_link_libraries(decan_core PRIVATE decan_document_processor)
file(TO_CMAKE_PATH "${DECAN_MODELS_DIR}" DECAN_MODELS_DIR_DEFINE)
target_compile_definitions(decan_core PRIVATE
    "DECAN_MODELS_DIR=\"${DECAN_MODELS_DIR_DEFINE}\""
)

# Keep model provisioning in the normal CMake dependency graph.  The script
# is intentionally dependency-free (it has an optional `onnx` implementation
# and a built-in protobuf encoder), so a clean build does not require torch or
# network access.  Existing files are preserved unless --force is requested;
# this makes it safe to point DECAN_MODELS_DIR at real project models.
set(DECAN_MODEL_FILES
    "${DECAN_MODELS_DIR}/ocr_printed.onnx"
    "${DECAN_MODELS_DIR}/ocr_handwriting.onnx"
    "${DECAN_MODELS_DIR}/seal_stamp_detector.onnx"
    "${DECAN_MODELS_DIR}/coat_of_arms_detector.onnx"
)
if(DECAN_SETUP_DUMMY_MODELS)
    add_custom_command(
        OUTPUT ${DECAN_MODEL_FILES}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DECAN_MODELS_DIR}"
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_models.py"
                --output-dir "${DECAN_MODELS_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_models.py"
        COMMENT "Provisioning deterministic ONNX test models"
        VERBATIM
    )
    add_custom_target(decan_setup_models DEPENDS ${DECAN_MODEL_FILES})
    add_dependencies(decan_core decan_setup_models)
endif()
