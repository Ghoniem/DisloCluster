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
# A private copy may live at <repo_root>/Libraries/sundials-7.1.1/; otherwise
# the system install is used.
#
# The repository root is found by walking UP to the .dislocluster_root marker,
# the same way dislocluster_code/paths.py does, rather than by counting "../"
# levels. Counting broke silently when ZrMicro moved from
# ZrClusterDynamics/ZrMicro/ to the repository root: the hint then pointed one
# level ABOVE the repository, and the build only kept working on machines that
# happened to have SUNDIALS installed system-wide.
set(_dc_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(DISLOCLUSTER_ROOT "")
foreach(_i RANGE 6)
    if(EXISTS "${_dc_dir}/.dislocluster_root")
        set(DISLOCLUSTER_ROOT "${_dc_dir}")
        break()
    endif()
    get_filename_component(_dc_dir "${_dc_dir}" DIRECTORY)
endforeach()

if(DISLOCLUSTER_ROOT)
    message(STATUS "DisloCluster root: ${DISLOCLUSTER_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH
         "${DISLOCLUSTER_ROOT}/Libraries/sundials-7.1.1")
else()
    message(WARNING
        "No .dislocluster_root marker found above ${CMAKE_CURRENT_SOURCE_DIR}; "
        "relying on a system SUNDIALS install.")
endif()

find_package(SUNDIALS QUIET)
if(NOT SUNDIALS_FOUND)
    message(FATAL_ERROR
        "SUNDIALS 7.1.1 was not found.\n"
        "  Looked in : ${DISLOCLUSTER_ROOT}/Libraries/sundials-7.1.1\n"
        "              and the default system locations.\n"
        "  Install it, or point CMake at it with\n"
        "      -DCMAKE_PREFIX_PATH=<path to the SUNDIALS install>\n"
        "  See README.md, 'Setup', step 2.")
endif()

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

# Link SUNDIALS — CVODES + ARKODE + linear solver modules (dense, band, SPGMR)
#
# CVODES rather than CVODE: solver.cpp already included <cvodes/cvodes.h> (the
# two share the CVodeCreate/CVodeInit/CVode symbol names, so linking plain cvode
# happened to work), but the reduced implicit block needs CVodeQuadInit /
# CVodeGetQuad / CVodeGetQuadNumRhsEvals, which exist only in CVODES.
target_link_libraries(solver
    SUNDIALS::cvodes
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

# ── Jacobian verification tool ───────────────────────────────────────────────
# Checks the forward-AD Jacobian against a central-difference Jacobian of the
# same templated core. Header-only: no SUNDIALS dependency.
add_executable(jac_check ${CMAKE_CURRENT_SOURCE_DIR}/jac_check.cpp)
target_include_directories(jac_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
