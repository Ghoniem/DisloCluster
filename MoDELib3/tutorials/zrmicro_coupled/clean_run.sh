#!/bin/bash
# Clean CD-only run of the ZrMicro <-> MoDELib (Zr3d_ghoniem) verification case.
#
#  1. Refresh the material file AND the mesh from Library/, LF-converted. Neither
#     is committed: they are generated copies. Both have already caused silent
#     failures here -- a stale Zr4.txt killed one run on a missing key, and a
#     CRLF .msh made the parser return ZERO nodes with no error at all.
#  2. Regenerate evl_0. That file is BOTH the initial config and the runID=0
#     output, so a bare re-run restarts from the previous run accumulated dose.
#     inputFiles/DD.txt pins startAtTimeStep=0 for the same reason.
#
# Output is written AFTER solve(), so evl_N holds the state at (N+1) dose steps.
#
# Paths are derived from this script's own location (tutorials/<case>/), so the
# case runs wherever the MoDELib checkout sits inside DisloCluster.
set -e
SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELIB="$(cd "$SIM/../.." && pwd)"
LIB="$MODELIB/Library"
BIN="${MODELIB_BUILD:-$MODELIB/build}/tools"
cd "$SIM"
test -x "$BIN/DDomp/DDomp" || {
    echo "DDomp not built at $BIN/DDomp/DDomp"
    echo "Build it with: bash $MODELIB/../Docs/Formulation/build_modelib.sh"
    exit 1
}
sed "s/\r$//" "$LIB/Materials/Zr3d_ghoniem.txt" > inputFiles/Zr3d_ghoniem.txt
sed "s/\r$//" "$LIB/Meshes/unitCube_15K.msh"    > inputFiles/unitCube_15K.msh
rm -rf evl F
mkdir -p evl F
# stat's size flag is spelled -c%s by GNU coreutils and -f%z by BSD/macOS;
# wc -c is the same number on both and needs no branch.
"$BIN/MicrostructureGenerator/microstructureGenerator" . > gen.log 2>&1
echo "GEN_EXIT=$? evl_0 bytes=$(wc -c < evl/evl_0.txt)"
"$BIN/DDomp/DDomp" . > run.log 2>&1
echo "DDOMP_EXIT=$?"
ls evl/ | head
