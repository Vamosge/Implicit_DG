cmake_minimum_required(VERSION 3.24)

project(gsx_dg_solver VERSION 1.0.0 LANGUAGES C CXX)

# Zig exposes its archiver as the `zig ar` subcommand. CMake's generic Clang
# platform module otherwise tries to execute the compiler as `zig qc ...`.
if(CMAKE_C_COMPILER MATCHES "[/\\\\]zig(\\.exe)?$")
  set(_gsx_zig_env
      "${CMAKE_COMMAND} -E env ZIG_GLOBAL_CACHE_DIR=${CMAKE_SOURCE_DIR}/build/.zig-cache/global ZIG_LOCAL_CACHE_DIR=${CMAKE_SOURCE_DIR}/build/.zig-cache/local")
  set(CMAKE_C_ARCHIVE_CREATE
      "${_gsx_zig_env} ${CMAKE_C_COMPILER} ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_FINISH
      "${_gsx_zig_env} ${CMAKE_C_COMPILER} ranlib <TARGET>")
  set(CMAKE_CXX_ARCHIVE_CREATE
      "${_gsx_zig_env} ${CMAKE_C_COMPILER} ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH
      "${_gsx_zig_env} ${CMAKE_C_COMPILER} ranlib <TARGET>")
endif()

option(GSX_USE_SYSTEM_CGNS "Use an already-installed CGNS package" OFF)
option(GSX_ENABLE_OPENMP "Enable OpenMP where available" ON)
option(GSX_REQUIRE_OPENMP "Fail configuration if OpenMP is unavailable" OFF)
set(GSX_CASE "bump89" CACHE STRING
    "Case target: bump89/bump177/bump353, plate35/plate69/plate137/plate273, naca113/naca113-original, naca113-familyII, or naca225")
set_property(CACHE GSX_CASE PROPERTY STRINGS
  bump89 bump177 bump353 plate35 plate69 plate137 plate273 flatplate
  naca113 naca113-original naca113-familyII naca225)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(GSX_USE_SYSTEM_CGNS)
  find_package(CGNS CONFIG REQUIRED)
else()
  set(CGNS_ENABLE_HDF5 OFF CACHE BOOL "" FORCE)
  set(CGNS_ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
  set(CGNS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(CGNS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/cgns EXCLUDE_FROM_ALL)
endif()

add_library(gsx_infrastructure STATIC
  src/cartesian_topology.cpp
  src/cgns_reader.cpp
  src/gmsh_reader.cpp
  src/wall_geometry.cpp
)
target_include_directories(gsx_infrastructure PUBLIC include)
target_compile_definitions(gsx_infrastructure PUBLIC DG_USE_CGNS=1)

if(TARGET CGNS::cgns-static)
  target_link_libraries(gsx_infrastructure PUBLIC CGNS::cgns-static)
elseif(TARGET CGNS::cgns-shared)
  target_link_libraries(gsx_infrastructure PUBLIC CGNS::cgns-shared)
elseif(TARGET cgns)
  target_link_libraries(gsx_infrastructure PUBLIC cgns)
else()
  message(FATAL_ERROR "No CGNS CMake target was found")
endif()

add_executable(gsx_dg
  src/main.cpp
  src/solver.cpp
)
target_link_libraries(gsx_dg PRIVATE gsx_infrastructure)
target_include_directories(gsx_dg PRIVATE include)

add_custom_target(benchmark
  COMMAND gsx_dg --benchmark
  DEPENDS gsx_dg
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL
  COMMENT "Running the fixed-size residual/Jacobian performance benchmark")

if(GSX_CASE STREQUAL "bump89")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_BUMP_CHANNEL=1 NX_CELLS=88 NY_CELLS=40
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/bump_struct_89x41_vol.cgns")
elseif(GSX_CASE STREQUAL "bump177")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_BUMP_CHANNEL=1 NX_CELLS=176 NY_CELLS=80
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/bump_struct_177x81_vol.cgns")
elseif(GSX_CASE STREQUAL "bump353")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_BUMP_CHANNEL=1 NX_CELLS=352 NY_CELLS=160
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/bump_struct_353x161_vol.cgns")
elseif(GSX_CASE STREQUAL "plate35" OR GSX_CASE STREQUAL "flatplate")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_FLAT_PLATE=1 NX_CELLS=34 NY_CELLS=24
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/grid_quad_35x25_vol.cgns")
elseif(GSX_CASE STREQUAL "plate69")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_FLAT_PLATE=1 NX_CELLS=68 NY_CELLS=48
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/grid_quad_69x49_vol.cgns")
elseif(GSX_CASE STREQUAL "plate137")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_FLAT_PLATE=1 NX_CELLS=136 NY_CELLS=96
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/grid_quad_137x97_vol.cgns")
elseif(GSX_CASE STREQUAL "plate273")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_FLAT_PLATE=1 NX_CELLS=272 NY_CELLS=192
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/grid_quad_273x193_vol.cgns")
elseif(GSX_CASE STREQUAL "naca113" OR GSX_CASE STREQUAL "naca113-original")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_NACA0012=1 DG_NACA_OFFICIAL_CGNS=1
    DG_NACA_GRID_FAMILY_ORIGINAL=1 DG_NACA_WAKE_PAIRS=24
    NX_CELLS=112 NY_CELLS=32
    CGNS_GRID_FILE="${CMAKE_SOURCE_DIR}/n0012_113-33.cgns"
    CGNS_GEOMETRY_GRID_FILE="${CMAKE_SOURCE_DIR}/n0012_449-129.cgns")
elseif(GSX_CASE STREQUAL "naca113-familyII")
  # NASA's Family-II levels are exactly nested. Read every other point from
  # the official 225x65 logical grid so this level and naca225 use one source.
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_NACA0012=1 DG_NACA_GRID_FAMILY_II=1 DG_NACA_WAKE_PAIRS=24
    DG_NACA_SOURCE_NI=225 DG_NACA_SOURCE_NJ=65
    NX_CELLS=112 NY_CELLS=32
    GMSH_GRID_FILE="${CMAKE_SOURCE_DIR}/naca0012_familyII_225x65.msh")
elseif(GSX_CASE STREQUAL "naca225")
  target_compile_definitions(gsx_dg PRIVATE
    DG_CASE_NACA0012=1 DG_NACA_GRID_FAMILY_II=1
    NX_CELLS=224 NY_CELLS=64
    GMSH_GRID_FILE="${CMAKE_SOURCE_DIR}/naca0012_familyII_225x65.msh")
else()
  message(FATAL_ERROR "Unknown GSX_CASE='${GSX_CASE}'")
endif()

if(MSVC)
  target_compile_options(gsx_dg PRIVATE /W4 /permissive- /O2 /Ob3)
else()
  target_compile_options(gsx_dg PRIVATE -Wall -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    target_compile_options(gsx_dg PRIVATE -O3)
  endif()
endif()

if(GSX_ENABLE_OPENMP)
  if(GSX_REQUIRE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
  else()
    find_package(OpenMP QUIET COMPONENTS CXX)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gsx_dg PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(gsx_dg PRIVATE DG_HAS_OPENMP=1)
    message(STATUS "GSX OpenMP enabled: ${OpenMP_CXX_VERSION}")
  else()
    target_compile_definitions(gsx_dg PRIVATE DG_HAS_OPENMP=0)
    message(STATUS "GSX OpenMP unavailable; building the serial solver")
  endif()
else()
  target_compile_definitions(gsx_dg PRIVATE DG_HAS_OPENMP=0)
  message(STATUS "GSX OpenMP disabled by GSX_ENABLE_OPENMP=OFF")
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME solver_cli_help COMMAND gsx_dg --help)
  add_test(NAME solver_cli_check_case COMMAND gsx_dg --check-case)
  set_tests_properties(solver_cli_check_case PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[CASE CHECK\\] status=ok")
  set(GSX_INITIAL_RESIDUAL_PREFIX "")
  if(MSVC)
    if(GSX_CASE STREQUAL "bump89")
      set(GSX_INITIAL_RESIDUAL_PREFIX "1.697089")
    elseif(GSX_CASE STREQUAL "plate35")
      set(GSX_INITIAL_RESIDUAL_PREFIX "1.633847")
    elseif(GSX_CASE STREQUAL "naca113" OR GSX_CASE STREQUAL "naca113-original")
      set(GSX_INITIAL_RESIDUAL_PREFIX "5.280486")
    elseif(GSX_CASE STREQUAL "naca113-familyII")
      set(GSX_INITIAL_RESIDUAL_PREFIX "8.319419")
    elseif(GSX_CASE STREQUAL "naca225")
      set(GSX_INITIAL_RESIDUAL_PREFIX "1.147836")
    endif()
  endif()
  if(GSX_INITIAL_RESIDUAL_PREFIX)
    add_test(NAME solver_cli_residual_check
      COMMAND "${CMAKE_COMMAND}"
        "-DSOLVER=$<TARGET_FILE:gsx_dg>"
        "-DEXPECTED_PREFIX=${GSX_INITIAL_RESIDUAL_PREFIX}"
        -P "${CMAKE_SOURCE_DIR}/tests/check_initial_residual.cmake")
  else()
    add_test(NAME solver_cli_residual_check COMMAND gsx_dg --residual-check)
    set_tests_properties(solver_cli_residual_check PROPERTIES
      PASS_REGULAR_EXPRESSION "\\[RESIDUAL CHECK\\] status=ok")
  endif()
  add_test(NAME solver_cli_invalid_max_steps COMMAND gsx_dg --max-steps 0)
  set_tests_properties(solver_cli_invalid_max_steps PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_conflicting_modes
    COMMAND gsx_dg --check-case --max-steps 1)
  set_tests_properties(solver_cli_conflicting_modes PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_jacobian_conflicting_modes
    COMMAND gsx_dg --jacobian-check --max-steps 1)
  set_tests_properties(solver_cli_jacobian_conflicting_modes PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME solver_cli_check_restart_conflict
    COMMAND gsx_dg --check-case --restart unused.bin)
  set_tests_properties(solver_cli_check_restart_conflict PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_residual_restart_conflict
    COMMAND gsx_dg --residual-check --restart unused.bin)
  set_tests_properties(solver_cli_residual_restart_conflict PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_cfl_max COMMAND gsx_dg --cfl-max 0)
  set_tests_properties(solver_cli_invalid_cfl_max PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_cfl_initial COMMAND gsx_dg --cfl-initial 0)
  set_tests_properties(solver_cli_invalid_cfl_initial PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_ns_tolerance COMMAND gsx_dg --ns-tolerance 0)
  set_tests_properties(solver_cli_invalid_ns_tolerance PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_sa_tolerance COMMAND gsx_dg --sa-tolerance 0)
  set_tests_properties(solver_cli_invalid_sa_tolerance PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_sa_cfl_scale COMMAND gsx_dg --sa-cfl-scale 0)
  set_tests_properties(solver_cli_invalid_sa_cfl_scale PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_sa_br2_factor COMMAND gsx_dg --sa-br2-factor 0)
  set_tests_properties(solver_cli_invalid_sa_br2_factor PROPERTIES WILL_FAIL TRUE)
  add_test(NAME solver_cli_q2_reset_requires_restart
    COMMAND gsx_dg --sa-q2-reset-on-restart --max-steps 1)
  set_tests_properties(solver_cli_q2_reset_requires_restart PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME solver_cli_freeze_ns_requires_restart
    COMMAND gsx_dg --freeze-ns --max-steps 1)
  set_tests_properties(solver_cli_freeze_ns_requires_restart PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME solver_cli_invalid_sa_limiter COMMAND gsx_dg --sa-limiter clip)
  set_tests_properties(solver_cli_invalid_sa_limiter PROPERTIES WILL_FAIL TRUE)

  add_executable(field_test tests/field_test.cpp)
  target_include_directories(field_test PRIVATE include)
  add_test(NAME field_storage COMMAND field_test)

  add_executable(wall_geometry_test tests/wall_geometry_test.cpp)
  target_link_libraries(wall_geometry_test PRIVATE gsx_infrastructure)
  add_test(NAME wall_geometry COMMAND wall_geometry_test)

  add_executable(cartesian_topology_test tests/cartesian_topology_test.cpp)
  target_link_libraries(cartesian_topology_test PRIVATE gsx_infrastructure)
  add_test(NAME cartesian_topology COMMAND cartesian_topology_test)

  add_executable(collectives_test tests/collectives_test.cpp)
  target_include_directories(collectives_test PRIVATE include)
  add_test(NAME serial_collectives COMMAND collectives_test)

  add_executable(stencil_coloring_test tests/stencil_coloring_test.cpp)
  target_include_directories(stencil_coloring_test PRIVATE include)
  add_test(NAME stencil_coloring COMMAND stencil_coloring_test)

  add_executable(reference_element_test tests/reference_element_test.cpp)
  target_include_directories(reference_element_test PRIVATE include)
  add_test(NAME reference_element COMMAND reference_element_test)

  add_executable(mapped_geometry_test tests/mapped_geometry_test.cpp)
  target_include_directories(mapped_geometry_test PRIVATE include)
  add_test(NAME mapped_geometry_cache COMMAND mapped_geometry_test)

  add_executable(modal_sensor_test tests/modal_sensor_test.cpp)
  target_include_directories(modal_sensor_test PRIVATE include)
  add_test(NAME modal_sensor COMMAND modal_sensor_test)

  add_executable(physics_test tests/physics_test.cpp)
  target_include_directories(physics_test PRIVATE include)
  add_test(NAME physics_kernels COMMAND physics_test)

  add_executable(viscous_sa_test tests/viscous_sa_test.cpp)
  target_include_directories(viscous_sa_test PRIVATE include)
  add_test(NAME viscous_sa_kernels COMMAND viscous_sa_test)

  add_executable(residual_workspace_test tests/residual_workspace_test.cpp)
  target_include_directories(residual_workspace_test PRIVATE include)
  add_test(NAME residual_workspace COMMAND residual_workspace_test)

  add_executable(forward_ad_example examples/forward_ad_example.cpp)
  add_test(NAME forward_ad_example COMMAND forward_ad_example)

  add_executable(block_ad_test tests/block_ad_test.cpp)
  target_include_directories(block_ad_test PRIVATE include)
  add_test(NAME block_ad_arithmetic COMMAND block_ad_test)

  add_executable(linear_solver_test tests/linear_solver_test.cpp)
  target_include_directories(linear_solver_test PRIVATE include)
  add_test(NAME linear_solver_module COMMAND linear_solver_test)

  add_executable(nonlinear_solver_test tests/nonlinear_solver_test.cpp)
  target_include_directories(nonlinear_solver_test PRIVATE include)
  add_test(NAME nonlinear_solver_policy COMMAND nonlinear_solver_test)

  add_executable(steady_driver_test tests/steady_driver_test.cpp)
  target_include_directories(steady_driver_test PRIVATE include)
  add_test(NAME steady_driver_policy COMMAND steady_driver_test)

  add_executable(diagnostics_test tests/diagnostics_test.cpp)
  target_include_directories(diagnostics_test PRIVATE include)
  add_test(NAME diagnostics_output COMMAND diagnostics_test)

  add_executable(ad_jacobian_test tests/ad_jacobian_test.cpp)
  target_include_directories(ad_jacobian_test PRIVATE include)
  add_test(NAME ad_jacobian_coloring COMMAND ad_jacobian_test)

  add_executable(ad_physics_test tests/ad_physics_test.cpp)
  target_include_directories(ad_physics_test PRIVATE include)
  add_test(NAME ad_physics_boundaries COMMAND ad_physics_test)

  add_executable(boundary_conditions_test tests/boundary_conditions_test.cpp)
  target_include_directories(boundary_conditions_test PRIVATE include)
  add_test(NAME boundary_condition_set COMMAND boundary_conditions_test)

  add_executable(boundary_physics_test tests/boundary_physics_test.cpp)
  target_include_directories(boundary_physics_test PRIVATE include)
  add_test(NAME boundary_characteristics COMMAND boundary_physics_test)

  add_executable(cgns_mesh_smoke tests/cgns_mesh_smoke.cpp)
  target_link_libraries(cgns_mesh_smoke PRIVATE gsx_infrastructure)
  add_test(NAME bump89_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/bump_struct_89x41_vol.cgns" 89 41 bump)
  add_test(NAME bump177_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/bump_struct_177x81_vol.cgns" 177 81 bump)
  add_test(NAME bump353_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/bump_struct_353x161_vol.cgns" 353 161 bump)
  add_test(NAME plate35_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/grid_quad_35x25_vol.cgns" 35 25 plate)
  add_test(NAME plate69_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/grid_quad_69x49_vol.cgns" 69 49 plate)
  add_test(NAME plate137_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/grid_quad_137x97_vol.cgns" 137 97 plate)
  add_test(NAME plate273_mesh
    COMMAND cgns_mesh_smoke "${CMAKE_SOURCE_DIR}/grid_quad_273x193_vol.cgns" 273 193 plate)

  add_executable(naca_cgns_mesh_test tests/naca_cgns_mesh_test.cpp)
  target_link_libraries(naca_cgns_mesh_test PRIVATE gsx_infrastructure)
  add_test(NAME naca113_nested_cgns_mesh
    COMMAND naca_cgns_mesh_test
      "${CMAKE_SOURCE_DIR}/n0012_113-33.cgns"
      "${CMAKE_SOURCE_DIR}/n0012_449-129.cgns")

  add_executable(gmsh_mesh_smoke tests/gmsh_mesh_smoke.cpp)
  target_link_libraries(gmsh_mesh_smoke PRIVATE gsx_infrastructure)
  add_test(NAME naca_familyII_nested_mesh
    COMMAND gmsh_mesh_smoke "${CMAKE_SOURCE_DIR}/naca0012_familyII_225x65.msh")
endif()
