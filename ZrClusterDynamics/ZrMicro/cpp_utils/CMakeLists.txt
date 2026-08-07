# ═════════════════════════════════════════════════════════════════════════════
#  ZrMicro C++ SUNDIALS solver build
#
#  Source layout (intentionally split across two folders):
#    ZrMicro/code/solver.cpp           — entry point (main): CLI parsing,
#                                          CVODE/ARKODE setup, stdout matrix
#    ZrMicro/cpp_utils/rate_equations.* — 12-equation RHS (this folder)
#    ZrMicro/cpp_utils/parameters.h     — Parameters struct
#
#  Build (run from this cpp_utils/ folder):
#    cmake -S . -B ../build -DCMAKE_BUILD_TYPE=Release
#    cmake --build ../build --config Release
# ═════════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.10)
project(zrmicro_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── SUNDIALS ─────────────────────────────────────────────────────────────────
# Expects the same Libraries/ layout as the Creep solver:
#   <repo_root>/Libraries/sundials-7.1.1/
get_filename_component(LIBRARIES_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../Libraries" ABSOLUTE)
list(PREPEND CMAKE_PREFIX_PATH "${LIBRARIES_DIR}/sundials-7.1.1")
find_package(SUNDIALS REQUIRED)

# ── OpenMP (optional) ────────────────────────────────────────────────────────
# Enables the parallel batch mode in solver.cpp (one thread per parameter case).
# The solver still builds and runs serially if OpenMP is unavailable.
find_package(OpenMP)

# ── Executable ───────────────────────────────────────────────────────────────
# solver.cpp lives in ZrMicro/code/; rate_equations.cpp lives here in cpp_utils/
add_executable(solver
    ../code/solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rate_equations.cpp
)

# Include cpp_utils/ so solver.cpp can find parameters.h and rate_equations.h
target_include_directories(solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link SUNDIALS — CVODE + ARKODE + linear solver modules (dense, band, SPGMR)
target_link_libraries(solver
    SUNDIALS::cvode
    SUNDIALS::arkode
    SUNDIALS::nvecserial
    SUNDIALS::sunlinsolband
    SUNDIALS::sunlinsolspgmr
)

# Link OpenMP when found (defines _OPENMP and adds /openmp on MSVC).
if(OpenMP_CXX_FOUND)
    target_link_libraries(solver OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for solver (parallel batch mode)")
else()
    message(STATUS "OpenMP NOT found — solver batch mode will run serially")
endif()
