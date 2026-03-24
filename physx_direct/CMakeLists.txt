cmake_minimum_required(VERSION 3.24)
project(softtissue_physx_direct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PHYSX_ROOT "${CMAKE_SOURCE_DIR}/third_party/PhysX/physx")
set(PHYSX_RELEASE_DIR "${CMAKE_SOURCE_DIR}/build/physx_sdk_vs/bin/win.x86_64.vc143.md/release")

if(NOT EXISTS "${PHYSX_ROOT}/include/PxPhysicsAPI.h")
  message(FATAL_ERROR "PhysX headers not found at ${PHYSX_ROOT}")
endif()

if(NOT EXISTS "${PHYSX_RELEASE_DIR}/PhysX.lib")
  message(FATAL_ERROR "Built PhysX release libraries not found at ${PHYSX_RELEASE_DIR}")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(st_physx_gm_reference
  app/main.cpp
)

target_include_directories(st_physx_gm_reference PRIVATE
  "${PHYSX_ROOT}/include"
  "${PHYSX_ROOT}/include/cudamanager"
)

target_compile_definitions(st_physx_gm_reference PRIVATE
  NOMINMAX
  WIN32
  WIN64
  _CRT_SECURE_NO_DEPRECATE
  _CRT_NONSTDC_NO_DEPRECATE
  _WINSOCK_DEPRECATED_NO_WARNINGS
  PX_PUBLIC_RELEASE=1
  PX_SUPPORT_PVD=0
  PX_SUPPORT_OMNI_PVD=0
)

if(MSVC)
  target_compile_options(st_physx_gm_reference PRIVATE /EHsc /bigobj)
endif()

target_link_libraries(st_physx_gm_reference PRIVATE
  "${PHYSX_RELEASE_DIR}/PhysX.lib"
  "${PHYSX_RELEASE_DIR}/PhysXCommon.lib"
  "${PHYSX_RELEASE_DIR}/PhysXFoundation.lib"
  "${PHYSX_RELEASE_DIR}/PhysXCooking.lib"
  "${PHYSX_RELEASE_DIR}/PhysXExtensions_static.lib"
  "${PHYSX_RELEASE_DIR}/PhysXCudaContextManager_static.lib"
  CUDA::cuda_driver
)

foreach(dll IN ITEMS PhysX.dll PhysXCommon.dll PhysXFoundation.dll PhysXCooking.dll PhysXGpu.dll)
  add_custom_command(TARGET st_physx_gm_reference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PHYSX_RELEASE_DIR}/${dll}"
      "$<TARGET_FILE_DIR:st_physx_gm_reference>/${dll}"
  )
endforeach()

add_custom_command(TARGET st_physx_gm_reference POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PHYSX_RELEASE_DIR}/PhysXGpu.dll"
    "$<TARGET_FILE_DIR:st_physx_gm_reference>/PhysXGpu_64.dll"
)
