#!/usr/bin/env bash
# Build MoDELib (DDomp + pyMoDELib), the 3-D half of DisloCluster, inside
# WSL/Ubuntu.
#
# Run this AFTER `wsl --install` + reboot + Ubuntu first-run setup, from a WSL
# shell:
#     bash <DisloCluster>/ZrClusterDynamics/Docs/Formulation/build_modelib_wsl.sh [MODELIB_ROOT]
#
# With no argument the MoDELib checkout is located relative to this script
# (../../../MoDELib3 = DisloCluster/MoDELib3), so the build works wherever the
# repository is cloned. The coupling notebook passes the path explicitly.
#
# It installs dependencies, patches the macOS-hardcoded bits of the repo's
# CMakeLists, skips the Qt-only DDqt tool, and builds the headless targets.
# The repo is read from the Windows filesystem via /mnt/<drive> by default.
set -euo pipefail

# <script>/../../..  == ZrClusterDynamics/Docs/Formulation -> ZrClusterDynamics -> DisloCluster
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/../../.." && pwd)"
REPO="${1:-$_REPO_ROOT/MoDELib3}"
echo "==> MoDELib repo: $REPO"
test -f "$REPO/CMakeLists.txt" || { echo "CMakeLists.txt not found in $REPO"; exit 1; }

echo "==> Installing dependencies (sudo apt-get)"
sudo apt-get update
# NOTE: libfftw3-dev is NOT optional despite CMakeLists.txt:65 warning that the
# noise generator is "disabled" when it is missing -- line 117 still does
#   target_link_libraries(MoDELib PUBLIC ${FFTW3_LIBRARIES} ...)
# unconditionally, so a missing FFTW3 puts the literal FFTW3_LIBRARIES-NOTFOUND
# on the link line and the CMake *generate* step fails (configure succeeds).
sudo apt-get install -y \
    build-essential cmake ninja-build \
    libeigen3-dev libomp-dev libfftw3-dev \
    python3-dev python3-pip pybind11-dev \
    libsuitesparse-dev libboost-all-dev

cd "$REPO"

# --- patch macOS-isms in CMakeLists.txt (idempotent) -------------------------
# 1) Eigen path (line ~10 hardcodes /opt/local/include/eigen3 and overrides -D)
sed -i 's|set(EIGEN3_INCLUDE_DIRS .*opt/local/include/eigen3.*)|set(EIGEN3_INCLUDE_DIRS /usr/include/eigen3)|' CMakeLists.txt
# 2) drop the macOS Qt prefix-path line (Qt is only needed by DDqt)
sed -i '\|Qt/.*/macos/lib/cmake|d' CMakeLists.txt
# 3) drop the MacPorts libgcc RPATH lines
sed -i '\|/opt/local/lib/libgcc|d' CMakeLists.txt

# --- skip the Qt-only DDqt tool ---------------------------------------------
if grep -q 'add_subdirectory.*DDqt' tools/CMakeLists.txt 2>/dev/null; then
    sed -i 's|^\(\s*\)add_subdirectory\(.*DDqt.*\)|\1# add_subdirectory\2  # disabled: needs Qt|' tools/CMakeLists.txt
fi

# --- discard a build cache created under a different absolute path -----------
# CMake hard-errors ("the current CMakeCache.txt directory ... is different")
# when a build tree is moved, which is exactly what happens after the MoDELib
# checkout is relocated into DisloCluster/. Detect it and start clean.
if [ -f build/CMakeCache.txt ]; then
    cached_src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt)"
    if [ -n "$cached_src" ] && [ "$cached_src" != "$REPO" ]; then
        echo "==> build/ was configured for $cached_src -- removing the stale cache"
        rm -rf build
    fi
fi

echo "==> Configuring (Release, pybind11 ON)"
# The repo declares cmake_minimum_required(VERSION 3.1.0); CMake >= 4.0 rejects
# anything below 3.5 outright. Ubuntu 24.04 ships CMake 3.28 (warning only), but
# a newer image would hard-fail here, so raise the floor explicitly. The flag is
# ignored by CMake < 3.31, so it is safe on either.
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DUSE_PYBIND11=ON \
    -DBUILD_TOOLS=ON

echo "==> Building (this can take 10-30 min)"
cmake --build build -j"$(nproc)"

echo "==> Done. Artifacts:"
ls -la build/tools/DDomp/DDomp 2>/dev/null && echo "  DDomp OK" || echo "  WARN: DDomp not found"
ls -la build/tools/pyMoDELib/pyMoDELib*.so 2>/dev/null && echo "  pyMoDELib OK" || echo "  (pyMoDELib optional)"
echo "==> py_utils/paths.py resolves MODELIB_BUILD = $REPO/build automatically."
echo "    Override with the MODELIB_ROOT / MODELIB_BUILD environment variables."
echo "    From Windows Python, run DDomp via WSL: pass wsl_exec=True to MoDELibFEMSolver."
