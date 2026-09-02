#!/usr/bin/env bash
# Build MoDELib (DDomp, microstructureGenerator and, when pybind11 is present,
# pyMoDELib) — the 3-D half of DisloCluster.
#
#     bash <DisloCluster>/Docs/Formulation/build_notes/build_modelib.sh [MODELIB_ROOT]
#
# Runs on Linux (native or WSL/Ubuntu) and on macOS (Intel or Apple silicon).
# With no argument the MoDELib checkout is located relative to this script
# (the .dislocluster_root marker above it), so the build works wherever the
# repository is cloned; dislocluster_code.build passes the path explicitly.
#
# Everything platform-specific is confined to install_dependencies() below.
# The CMakeLists themselves are portable: they search for each dependency,
# probe every optimization flag against the compiler in hand, and skip what is
# absent. This script used to `sed` the macOS-isms out of CMakeLists.txt and
# comment out the Qt tool on every run, which is why it could only ever produce
# a build for the platform it was last run on.
set -euo pipefail

# Walk up to the .dislocluster_root marker, the way dislocluster_code/paths.py
# does, rather than counting ".." levels -- this script has already moved once
# (Docs/Formulation -> Docs/Formulation/build_notes) and a hard-coded depth
# silently resolved the root to Docs/, so the default MoDELib checkout was
# looked for at Docs/MoDELib3 and Eigen would have unpacked into Docs/Libraries.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$_SCRIPT_DIR"
while [ ! -e "$_REPO_ROOT/.dislocluster_root" ] && [ "$_REPO_ROOT" != "/" ]; do
    _REPO_ROOT="$(dirname "$_REPO_ROOT")"
done
if [ ! -e "$_REPO_ROOT/.dislocluster_root" ]; then
    echo "could not find .dislocluster_root above $_SCRIPT_DIR" >&2
    exit 1
fi
REPO="${1:-$_REPO_ROOT/MoDELib3}"
BUILD="${MODELIB_BUILD:-$REPO/build}"

echo "==> MoDELib repo : $REPO"
echo "==> build dir    : $BUILD"
test -f "$REPO/CMakeLists.txt" || { echo "CMakeLists.txt not found in $REPO"; exit 1; }

# ── how many cores, and which platform ───────────────────────────────────────
# nproc is GNU coreutils and does not exist on macOS; sysctl is macOS-only.
if command -v nproc >/dev/null 2>&1; then
    NPROC="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    NPROC="$(sysctl -n hw.ncpu)"
else
    NPROC=4
fi

OS="$(uname -s)"

# `sudo` is absent from minimal containers and unnecessary when already root.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# ── dependencies ─────────────────────────────────────────────────────────────
# Required : a C++20 compiler, CMake, Eigen 3.
# Optional : OpenMP (threading), FFTW3 + Boost (glide-plane noise), SuiteSparse
#            (faster sparse solvers), pybind11 (pyMoDELib).
#
# NOTE on FFTW: the CMakeLists warns that the noise generator is "disabled"
# without it, but it is worth installing -- MoDELib links it into the library
# whenever it is found, and the two sides of a comparison should agree on which
# code paths are compiled in.
install_dependencies() {
    if [ "${SKIP_DEPS:-0}" = "1" ]; then
        echo "==> SKIP_DEPS=1 — not installing anything"
        return
    fi
    case "$OS" in
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            echo "==> Installing dependencies (apt-get)"
            $SUDO apt-get update
            $SUDO apt-get install -y \
                build-essential cmake ninja-build \
                libeigen3-dev libomp-dev libfftw3-dev \
                python3-dev python3-pip pybind11-dev \
                libsuitesparse-dev libboost-all-dev
        elif command -v dnf >/dev/null 2>&1; then
            echo "==> Installing dependencies (dnf)"
            $SUDO dnf install -y gcc-c++ cmake ninja-build eigen3-devel \
                fftw-devel suitesparse-devel boost-devel \
                python3-devel pybind11-devel libomp-devel
        elif command -v pacman >/dev/null 2>&1; then
            echo "==> Installing dependencies (pacman)"
            $SUDO pacman -Sy --noconfirm base-devel cmake ninja eigen fftw \
                suitesparse boost python pybind11 openmp
        else
            echo "==> No apt/dnf/pacman found — install the dependencies yourself:"
            echo "    a C++20 compiler, CMake >= 3.16, Eigen 3, and optionally"
            echo "    FFTW3, Boost, SuiteSparse, OpenMP, pybind11."
        fi
        ;;
    Darwin)
        if command -v brew >/dev/null 2>&1; then
            echo "==> Installing dependencies (Homebrew)"
            # Homebrew is a no-op for what is already installed.
            brew install cmake ninja eigen fftw suite-sparse boost libomp
        else
            echo "==> Homebrew not found. Install it from https://brew.sh, then:"
            echo "    brew install cmake ninja eigen fftw suite-sparse boost libomp"
            echo "  (or install the same libraries by hand and re-run with SKIP_DEPS=1)"
        fi
        # Command Line Tools supply clang; without them there is no compiler at all.
        if ! xcode-select -p >/dev/null 2>&1; then
            echo "==> Xcode Command Line Tools missing — run: xcode-select --install"
            exit 1
        fi
        ;;
    *)
        echo "==> Unrecognized platform '$OS' — skipping dependency installation."
        echo "    Install a C++20 compiler, CMake >= 3.16 and Eigen 3, then re-run"
        echo "    with SKIP_DEPS=1."
        ;;
    esac
}
install_dependencies

# ── Eigen 3, specifically ────────────────────────────────────────────────────
# MoDELib is written against Eigen 3 and misbehaves on anything newer: built
# against Eigen 5 it compiles, then the BiCGSTAB solve inside the mobile Newton
# iteration breaks down on the first step ("Iterative FixedDirichletSolver
# failed") on a case that converges with 3.4.0. Debian/Fedora/Arch still ship
# 3.4.x as libeigen3-dev, but Homebrew's `eigen` formula moved to 5.x, so on
# macOS a private 3.4.0 is fetched into <repo>/Libraries/ -- the same place the
# 0-D solver's private SUNDIALS lives. CMakeLists.txt prefers it automatically.
EIGEN_VERSION=3.4.0
ensure_eigen3() {
    local m
    for m in /usr/include/eigen3 /usr/local/include/eigen3 \
             "$(command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null)/include/eigen3" \
             /opt/local/include/eigen3; do
        [ -f "$m/Eigen/src/Core/util/Macros.h" ] || continue
        if grep -q "define EIGEN_WORLD_VERSION 3" "$m/Eigen/src/Core/util/Macros.h"; then
            echo "==> Eigen 3 found: $m"
            return 0
        fi
    done

    local dest="$_REPO_ROOT/Libraries/eigen-$EIGEN_VERSION"
    if [ -f "$dest/Eigen/src/Core/util/Macros.h" ]; then
        echo "==> Eigen 3 found: $dest"
        return 0
    fi

    echo "==> No Eigen 3 on this system (Homebrew's eigen is 5.x) — fetching $EIGEN_VERSION"
    mkdir -p "$_REPO_ROOT/Libraries"
    local url="https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz"
    local tgz="$_REPO_ROOT/Libraries/eigen-$EIGEN_VERSION.tar.gz"
    if command -v curl >/dev/null 2>&1; then
        curl -sSL -o "$tgz" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$tgz" "$url"
    else
        echo "    neither curl nor wget is available — download $url by hand,"
        echo "    unpack it into $_REPO_ROOT/Libraries/, and re-run."
        exit 1
    fi
    tar xzf "$tgz" -C "$_REPO_ROOT/Libraries"
    rm -f "$tgz"
    test -f "$dest/Eigen/src/Core/util/Macros.h" || {
        echo "    the download did not unpack into $dest"; exit 1; }
    echo "==> Eigen $EIGEN_VERSION installed at $dest (headers only, git-ignored)"
}
[ "${SKIP_DEPS:-0}" = "1" ] || ensure_eigen3

cd "$REPO"

# ── discard a build cache created under a different absolute path ─────────────
# CMake hard-errors ("the current CMakeCache.txt directory ... is different")
# when a build tree is moved, which is exactly what happens when the checkout is
# relocated, copied to another machine, or reached through a different mount
# (/mnt/d from WSL vs D:\ from Windows). Detect it and start clean.
if [ -f "$BUILD/CMakeCache.txt" ]; then
    cached_src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")"
    if [ -n "$cached_src" ] && [ "$cached_src" != "$REPO" ]; then
        echo "==> $BUILD was configured for $cached_src — removing the stale cache"
        rm -rf "$BUILD"
    fi
fi

# Ninja when available, otherwise whatever CMake defaults to (Make).
GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

echo "==> Configuring (Release)"
cmake -S . -B "$BUILD" "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release

echo "==> Building with $NPROC jobs (this can take 10-30 min)"
cmake --build "$BUILD" -j"$NPROC"

# ── report ───────────────────────────────────────────────────────────────────
# Exit status alone is not enough: report what actually exists on disk, since
# dislocluster_code.build verifies the artifact rather than the return code.
echo "==> Done. Artifacts:"
status=0
if [ -x "$BUILD/tools/DDomp/DDomp" ]; then
    echo "  DDomp OK                  $BUILD/tools/DDomp/DDomp"
else
    echo "  ERROR: DDomp was not produced"
    status=1
fi
[ -x "$BUILD/tools/MicrostructureGenerator/microstructureGenerator" ] \
    && echo "  microstructureGenerator OK"
ls "$BUILD"/tools/pyMoDELib/pyMoDELib*.so >/dev/null 2>&1 \
    && echo "  pyMoDELib OK (optional)" \
    || echo "  pyMoDELib not built (optional: needs pybind11)"

echo "==> dislocluster_code/paths.py resolves MODELIB_BUILD = $BUILD automatically."
echo "    Override with the MODELIB_ROOT / MODELIB_BUILD environment variables."
exit $status
