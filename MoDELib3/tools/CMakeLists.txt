
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/MicrostructureGenerator MicrostructureGenerator)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DDomp DDomp)

# DDqt is the GUI: it needs Qt6 and VTK >= 9.4, which no headless machine has
# and which the coupling never uses. Detect them rather than hard-coding the
# decision -- this line used to be commented out by a `sed` in the WSL build
# script, so a machine that *did* have Qt could not build the GUI without
# editing the file back, and re-running the script re-commented it.
find_package(Qt6 COMPONENTS Widgets OpenGL OpenGLWidgets QUIET)
find_package(VTK 9.4.0 QUIET)
if(Qt6_FOUND AND VTK_FOUND)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DDqt DDqt)
else()
	message(STATUS "Qt6/VTK not found -- the DDqt GUI is not built")
endif()

# pybind11_FOUND, not USE_PYBIND11: the option only asks for pyMoDELib, and
# adding the subdirectory without pybind11 fails at configure time on
# `pybind11_add_module` -- an unknown command, so the whole build stops, DDomp
# included, on any machine that has no pybind11.
if(pybind11_FOUND)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pyMoDELib pyMoDELib)
endif()
