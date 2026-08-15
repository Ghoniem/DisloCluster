# ═════════════════════════════════════════════════════════════════════════════
#  MoDELib build — Linux (native or WSL), macOS (Intel or Apple silicon).
#
#  Nothing below hard-codes a path, a compiler or an instruction set: every
#  dependency is searched for, every optimization flag is tested against the
#  compiler in hand before it is used, and anything optional that is absent
#  degrades to a message rather than to a broken build. The configuration
#  summary at the bottom prints what was actually found.
#
#  Build it with Docs/Formulation/build_modelib.sh, which installs the
#  dependencies for the platform first. By hand:
#      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# ═════════════════════════════════════════════════════════════════════════════

# 3.16 rather than 3.1.0: CMake >= 4.0 refuses to run a project that asks for
# anything below 3.5, so the old floor made the build fail outright on a
# current CMake. 3.16 is what Ubuntu 20.04 shipped -- older than any machine
# this is built on.
cmake_minimum_required(VERSION 3.16)
project(MoDELib VERSION 2.0.0 LANGUAGES C CXX)

include(CheckCXXCompilerFlag)

option(USE_PYBIND11 "enables pybind11" ON)
option(BUILD_TOOLS "builds tools" ON)
option(USE_SUITESPARSE "use SuiteSparse (CHOLMOD/UMFPACK) solvers when present" ON)
option(USE_FAST_MATH "allow -Ofast (non-IEEE floating point)" ON)
option(USE_NATIVE_ARCH "tune the build for the host CPU (not portable between machines)" ON)
list(APPEND CMAKE_BUILD_RPATH ${RPATH})

# ── Homebrew / MacPorts prefixes ─────────────────────────────────────────────
# On macOS the dependencies live under a package-manager prefix that differs by
# architecture (/opt/homebrew on Apple silicon, /usr/local on Intel, /opt/local
# under MacPorts). Ask the package manager rather than guessing, and fall back
# to the conventional locations when it is not installed.
if(APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(COMMAND ${BREW_EXECUTABLE} --prefix
                        OUTPUT_VARIABLE HOMEBREW_PREFIX
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    foreach(_p ${HOMEBREW_PREFIX} /opt/homebrew /usr/local /opt/local)
        if(IS_DIRECTORY "${_p}")
            list(APPEND CMAKE_PREFIX_PATH "${_p}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif()

# Default to Release, but let -DCMAKE_BUILD_TYPE=Debug through: this used to be
# an unconditional set(), which silently discarded whatever the caller asked for.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization and warning flags ───────────────────────────────────────────
# Each flag is probed before use. -march=native is rejected outright by Apple
# clang on arm64 ("unsupported argument 'native'"), and -fopenmp is rejected by
# Apple clang everywhere -- both were unconditional here, so the build could not
# get past its first translation unit on a Mac. OpenMP is handled by
# find_package(OpenMP) below instead of by a raw flag.
set(_dc_flags -Wall -Wextra -Wno-unused-variable -Wno-unused-but-set-variable
              -Wno-unused-parameter)
if(USE_NATIVE_ARCH)
    list(APPEND _dc_flags -march=native)     # Apple silicon: falls back to -mcpu below
endif()
if(USE_FAST_MATH)
    list(APPEND _dc_flags -Ofast)
endif()
foreach(_flag IN LISTS _dc_flags)
    string(MAKE_C_IDENTIFIER "HAVE${_flag}" _flag_ok)
    check_cxx_compiler_flag("${_flag}" ${_flag_ok})
    if(${_flag_ok})
        string(APPEND CMAKE_CXX_FLAGS " ${_flag}")
    endif()
endforeach()
# 103 files under include/ and src/ call assert() without including <cassert>:
# GNU libstdc++ pulls it in transitively, libc++ (every clang on macOS) does
# not, so the same source stops compiling at the first translation unit. This
# force-include is the whole-tree substitute for editing all 103 files, which
# would put this fork permanently at odds with upstream. It is a no-op where
# the header was already reachable.
check_cxx_compiler_flag("-include cassert" HAVE_force_include)
if(HAVE_force_include)
    string(APPEND CMAKE_CXX_FLAGS " -include cassert")
endif()

if(USE_NATIVE_ARCH AND NOT HAVE_march_native)
    # Apple clang on arm64 spells host tuning -mcpu=apple-m1 (and defaults to
    # the host CPU anyway); on anything else the loss is only tuning.
    check_cxx_compiler_flag("-mcpu=apple-m1" HAVE_mcpu_apple_m1)
    if(HAVE_mcpu_apple_m1)
        string(APPEND CMAKE_CXX_FLAGS " -mcpu=apple-m1")
    endif()
endif()

### OpenMP ###
# MoDELib parallelizes with `#pragma omp`; without OpenMP it still compiles and
# runs, serially. Apple clang needs to be pointed at Homebrew's libomp, which
# FindOpenMP does not do on its own.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE AND BREW_EXECUTABLE)
    execute_process(COMMAND ${BREW_EXECUTABLE} --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/include/omp.h")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES omp)
        # Not REQUIRED: OpenMP is optional here, and a missing runtime should
        # cost threading, not the whole build.
        find_library(OpenMP_omp_LIBRARY omp HINTS "${LIBOMP_PREFIX}/lib")
        if(OpenMP_omp_LIBRARY)
            find_package(OpenMP QUIET)
        endif()
    endif()
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP_CXX_VERSION:" ${OpenMP_CXX_VERSION})
else()
    message(WARNING "OpenMP not found -- MoDELib will run single-threaded")
endif()

### Eigen ###
# Header-only, and the one hard requirement. MoDELib is written against Eigen 3
# and ONLY Eigen 3 -- see the version check below, which is not a formality.
#
# -DEIGEN3_INCLUDE_DIRS=<dir> wins; then a copy under <repo>/Libraries/eigen*,
# the way SUNDIALS is kept for the 0-D solver; then Eigen3Config.cmake; then a
# directory search. This line used to be a bare `set(... /usr/include/eigen3)`,
# which is where Debian puts Eigen and nowhere else -- so it both overrode the
# caller's -D and pointed at a non-existent directory on every other platform.
#
# The repository root is found by walking up to the .dislocluster_root marker,
# the same way dislocluster_code/paths.py and ZrMicro's CMakeLists do.
set(_dc_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(DISLOCLUSTER_ROOT "")
foreach(_i RANGE 6)
    if(EXISTS "${_dc_dir}/.dislocluster_root")
        set(DISLOCLUSTER_ROOT "${_dc_dir}")
        break()
    endif()
    get_filename_component(_dc_dir "${_dc_dir}" DIRECTORY)
endforeach()

# Eigen states its version in two different places depending on the era:
# <dir>/Eigen/src/Core/util/Macros.h up to 3.4.x, and <dir>/Eigen/Version from
# 5.0.0 on. Read both, or a version check written against one of them silently
# passes on the other -- which is exactly what an earlier revision of this file
# did: Eigen 5 keeps EIGEN_WORLD_VERSION at 3 ("forever ... for the Eigen3
# library"), so it reported itself as 3.5.0 and sailed through a `^3\.` test.
function(dc_eigen_version dir out_var)
    set(${out_var} "" PARENT_SCOPE)
    foreach(_f "${dir}/Eigen/Version" "${dir}/Eigen/src/Core/util/Macros.h")
        if(NOT EXISTS "${_f}")
            continue()
        endif()
        file(STRINGS "${_f}" _lines
             REGEX "#define EIGEN_(WORLD|MAJOR|MINOR)_VERSION")
        set(_v "")
        foreach(_key WORLD MAJOR MINOR)
            foreach(_l ${_lines})
                if(_l MATCHES "#define EIGEN_${_key}_VERSION[ \t]+([0-9]+)")
                    list(APPEND _v ${CMAKE_MATCH_1})
                endif()
            endforeach()
        endforeach()
        list(LENGTH _v _n)
        if(_n EQUAL 3)
            list(JOIN _v "." _joined)
            set(${out_var} "${_joined}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

if(NOT EIGEN3_INCLUDE_DIRS AND DISLOCLUSTER_ROOT)
    file(GLOB _dc_eigen_dirs "${DISLOCLUSTER_ROOT}/Libraries/eigen*")
    foreach(_d ${_dc_eigen_dirs})
        if(EXISTS "${_d}/Eigen/src/Core/util/Macros.h")
            set(EIGEN3_INCLUDE_DIRS "${_d}")
            break()
        endif()
    endforeach()
endif()
if(NOT EIGEN3_INCLUDE_DIRS)
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(EIGEN3_INCLUDE_DIR)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
    elseif(TARGET Eigen3::Eigen)
        get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen
                            INTERFACE_INCLUDE_DIRECTORIES)
    endif()
endif()
if(NOT EIGEN3_INCLUDE_DIRS)
    find_path(EIGEN3_INCLUDE_DIRS signature_of_eigen3_matrix_library
              HINTS ${CMAKE_PREFIX_PATH} /usr /usr/local /opt/homebrew /opt/local
              PATH_SUFFIXES include/eigen3 eigen3 include)
endif()
if(NOT EIGEN3_INCLUDE_DIRS)
    message(FATAL_ERROR
        "Eigen 3 was not found.\n"
        "  Install it   : sudo apt-get install libeigen3-dev  (3.4.0)\n"
        "  Or let Docs/Formulation/build_modelib.sh fetch 3.4.0 into "
        "<repo>/Libraries/\n"
        "  Or point at it: -DEIGEN3_INCLUDE_DIRS=<dir containing the Eigen/ folder>")
endif()

# Eigen 3, not "whatever is installed". Homebrew's `eigen` formula is 5.x now,
# and MoDELib built against it compiles but does not WORK: the BiCGSTAB solve
# in ClusterDynamicsFEM's mobile Newton iteration breaks down on the first step
# ("Iterative FixedDirichletSolver failed") on a case that converges with 3.4.0.
# A silent numerical regression is worse than a build error, so this is fatal.
# The bound is on WORLD.MAJOR, i.e. 3.4.x and older: Eigen's own "5.0.1" is
# spelled 3.5.0 by these macros. -DEIGEN3_ALLOW_UNTESTED=ON overrides, for
# whoever wants to find out whether a newer Eigen has fixed the breakdown.
dc_eigen_version("${EIGEN3_INCLUDE_DIRS}" EIGEN3_VERSION_STRING)
if(EIGEN3_VERSION_STRING MATCHES "^([0-9]+)\\.([0-9]+)\\.")
    set(_eig_world ${CMAKE_MATCH_1})
    set(_eig_major ${CMAKE_MATCH_2})
else()
    set(_eig_world 0)
    set(_eig_major 0)
endif()
if(NOT EIGEN3_ALLOW_UNTESTED
   AND EIGEN3_VERSION_STRING
   AND (NOT _eig_world EQUAL 3 OR _eig_major GREATER 4))
    message(FATAL_ERROR
        "Eigen ${EIGEN3_VERSION_STRING} was found at ${EIGEN3_INCLUDE_DIRS}, but "
        "MoDELib requires 3.4.x or older (3.4.0 is what it is developed "
        "against).\n"
        "  Those macros spell the 5.0.1 release '3.5.0': Eigen keeps "
        "EIGEN_WORLD_VERSION at 3 and moved the release number into "
        "EIGEN_MAJOR_VERSION.\n"
        "  Homebrew's `eigen` formula is 5.x: `brew install eigen` is NOT enough "
        "on macOS.\n"
        "  Docs/Formulation/build_modelib.sh fetches 3.4.0 into "
        "<repo>/Libraries/eigen-3.4.0 and this file prefers it automatically.\n"
        "  Or point at your own copy: -DEIGEN3_INCLUDE_DIRS=<dir>")
endif()
message(STATUS "EIGEN3_INCLUDE_DIRS:" ${EIGEN3_INCLUDE_DIRS}
               " (version ${EIGEN3_VERSION_STRING})")

### Boost ###
# Headers only, and only for the glide-plane noise generator. CMake 3.30
# deprecated its own FindBoost in favour of the config files Boost installs, so
# ask for the config first and let the module be the fallback on older CMake.
if(POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost QUIET CONFIG)
if(NOT Boost_FOUND AND NOT POLICY CMP0167)
	find_package(Boost QUIET)
endif()
if(Boost_FOUND)
	message(STATUS "Boost_VERSION:" ${Boost_VERSION})
	include_directories(${Boost_INCLUDE_DIRS})
else()
	message(STATUS "Boost not found -- the glide-plane noise generator is disabled")
endif()

### SuiteSparse ###
# Optional: CHOLMOD/UMFPACK only swap Eigen's sparse LLT/LU for the SuiteSparse
# ones (ClusterDynamicsFEM.h), so the build is correct either way. SuiteSparse 7
# ships its own config files but they are found only when the package-manager
# prefix is on CMAKE_PREFIX_PATH -- which is what the APPLE block above puts
# there. The modern configs export targets and leave *_LIBRARIES empty, so the
# target is preferred and the variable is a fallback: linking the empty variable
# would define MODELIB_CHOLMOD without linking anything and fail at link time.
set(SUITESPARSE_LIBRARIES "")
if(USE_SUITESPARSE)
	find_package(CHOLMOD QUIET)
	if(TARGET SuiteSparse::CHOLMOD)
		set(CHOLMOD_LIBRARIES SuiteSparse::CHOLMOD)
	endif()
	if(CHOLMOD_FOUND AND CHOLMOD_LIBRARIES)
		message(STATUS "CHOLMOD_VERSION:" ${CHOLMOD_VERSION})
		message(STATUS "CHOLMOD_LIBRARIES:" ${CHOLMOD_LIBRARIES})
		if(CHOLMOD_INCLUDE_DIR)
			include_directories(${CHOLMOD_INCLUDE_DIR})
		endif()
		list(APPEND SUITESPARSE_LIBRARIES ${CHOLMOD_LIBRARIES})
		add_definitions(-D MODELIB_CHOLMOD)
	else()
		set(CHOLMOD_FOUND FALSE)
		message(STATUS "CHOLMOD not found -- using Eigen's SimplicialLLT")
	endif()

	find_package(UMFPACK QUIET)
	if(TARGET SuiteSparse::UMFPACK)
		set(UMFPACK_LIBRARIES SuiteSparse::UMFPACK)
	endif()
	if(UMFPACK_FOUND AND UMFPACK_LIBRARIES)
		message(STATUS "UMFPACK_VERSION:" ${UMFPACK_VERSION})
		message(STATUS "UMFPACK_LIBRARIES:" ${UMFPACK_LIBRARIES})
		if(UMFPACK_INCLUDE_DIR)
			include_directories(${UMFPACK_INCLUDE_DIR})
		endif()
		list(APPEND SUITESPARSE_LIBRARIES ${UMFPACK_LIBRARIES})
		add_definitions(-D MODELIB_UMFPACK)
	else()
		set(UMFPACK_FOUND FALSE)
		message(STATUS "UMFPACK not found -- using Eigen's SparseLU")
	endif()
endif()

### fftw3 ### find_package is currently broken. Workaround using find_library
# HINTS so the Homebrew/MacPorts prefix is searched: find_library does not
# consult CMAKE_PREFIX_PATH entries that are not standard system directories.
find_library(FFTW3_LIBRARIES fftw3
             HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
if(FFTW3_LIBRARIES)
	get_filename_component(FFTW3_LIBRARY_DIRS "${FFTW3_LIBRARIES}" DIRECTORY)
	set(FFTW3_INCLUDE_DIRS ${FFTW3_LIBRARY_DIRS}/../include)
	message(STATUS "FFTW3_LIBRARIES:" ${FFTW3_LIBRARIES})
	message(STATUS "FFTW3_LIBRARY_DIRS:" ${FFTW3_LIBRARY_DIRS})
	message(STATUS "FFTW3_INCLUDE_DIRS:" ${FFTW3_INCLUDE_DIRS})
	include_directories(${Boost_INCLUDE_DIRS})
	include_directories(${FFTW3_INCLUDE_DIRS})
	add_definitions(-D _MODEL_GLIDE_PLANE_NOISE_GENERATOR_)
else()
	# The link line below is conditional on this variable for a reason: it used
	# to interpolate FFTW3_LIBRARIES unconditionally, so a missing FFTW put the
	# literal "FFTW3_LIBRARIES-NOTFOUND" on the link line and CMake's *generate*
	# step failed -- after configure had reported success.
	set(FFTW3_LIBRARIES "")
	message(STATUS "fftw3 not found -- the glide-plane noise generator is disabled")
endif()

### pybind11 ###
# Optional. pyMoDELib is used by legacy/modelib_fem.py only; the coupling drives
# DDomp as a subprocess, so a machine without pybind11 still gets everything the
# march needs. This was `REQUIRED`, which turned a missing optional dependency
# into a hard configure failure.
if(USE_PYBIND11)
	cmake_policy(SET CMP0057 NEW)
	find_package(Python COMPONENTS Interpreter Development QUIET)
	if(Python_FOUND)
		message(STATUS "Python_VERSION:" ${Python_VERSION})
		# pybind11 is most often installed as a Python package rather than by
		# the system package manager; ask the interpreter where its CMake
		# config is before giving up.
		execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
		                OUTPUT_VARIABLE _pybind11_cmakedir
		                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
		find_package(pybind11 CONFIG QUIET HINTS ${_pybind11_cmakedir})
	else()
		message(WARNING "Python not found, pybind11 is disabled")
	endif()
endif()

if(pybind11_FOUND)
	message(STATUS "pybind11_VERSION:" ${pybind11_VERSION})
	set(pybind11_LIBRARIES pybind11::embed pybind11::headers pybind11::module)
	message(STATUS "pybind11_LIBRARIES: ${pybind11_LIBRARIES}")
	add_definitions(-D_MODEL_PYBIND11_)
	Python_add_library(MoDELib SHARED "")
else()
	message(STATUS "pybind11 not found -- pyMoDELib is disabled (DDomp is not)")
	add_library(MoDELib "")
endif()

set(MoDELib_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories ( ${EIGEN3_INCLUDE_DIRS} )
include_directories(${MoDELib_INCLUDE_DIR}/Utilities/)
include_directories(${MoDELib_INCLUDE_DIR}/IO/)
include_directories(${MoDELib_INCLUDE_DIR}/Math/)
include_directories(${MoDELib_INCLUDE_DIR}/Geometry/)
include_directories(${MoDELib_INCLUDE_DIR}/Mesh/)
include_directories(${MoDELib_INCLUDE_DIR}/Lattices/)
include_directories(${MoDELib_INCLUDE_DIR}/FEM/)
include_directories(${MoDELib_INCLUDE_DIR}/GlidePlanes/)
include_directories(${MoDELib_INCLUDE_DIR}/Quadrature/)
include_directories(${MoDELib_INCLUDE_DIR}/PolycrystallineMaterials/)
include_directories(${MoDELib_INCLUDE_DIR}/DislocationDynamicsIO/)
include_directories(${MoDELib_INCLUDE_DIR}/DislocationMobilities/)
include_directories(${MoDELib_INCLUDE_DIR}/DislocationDynamicsBase/)
include_directories(${MoDELib_INCLUDE_DIR}/DislocationMicrostructure/)
include_directories(${MoDELib_INCLUDE_DIR}/LoopNetwork/)
include_directories(${MoDELib_INCLUDE_DIR}/ParticleInteraction/)
include_directories(${MoDELib_INCLUDE_DIR}/SpaceDecomposition/)
include_directories(${MoDELib_INCLUDE_DIR}/DiscreteCrackMechanics/)
include_directories(${MoDELib_INCLUDE_DIR}/ClusterDynamics/)
include_directories(${MoDELib_INCLUDE_DIR}/ElasticDeformation/)
include_directories(${MoDELib_INCLUDE_DIR}/InclusionMicrostructure/)
include_directories(${MoDELib_INCLUDE_DIR}/DislocationDynamics/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ src)
target_link_libraries(MoDELib PUBLIC ${FFTW3_LIBRARIES} ${pybind11_LIBRARIES} ${SUITESPARSE_LIBRARIES})
if(OpenMP_CXX_FOUND)
	target_link_libraries(MoDELib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BUILD_TOOLS)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools tools)
else()
endif()

# ── Configuration summary ────────────────────────────────────────────────────
# What is optional here differs from machine to machine, and the difference is
# worth seeing at a glance: SuiteSparse swaps the sparse solvers, OpenMP decides
# whether the CD assembly is threaded.
message(STATUS "──────── MoDELib configuration ────────")
message(STATUS "  system      : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags   : ${CMAKE_CXX_FLAGS}")
message(STATUS "  Eigen       : ${EIGEN3_INCLUDE_DIRS}")
foreach(_dep OpenMP_CXX Boost CHOLMOD UMFPACK pybind11)
	if(${_dep}_FOUND)
		message(STATUS "  ${_dep}: yes")
	else()
		message(STATUS "  ${_dep}: no")
	endif()
endforeach()
if(FFTW3_LIBRARIES)
	message(STATUS "  FFTW3: yes")
else()
	message(STATUS "  FFTW3: no")
endif()
message(STATUS "───────────────────────────────────────")
