"""Moved to :mod:`dislocluster_code.zerod.reaction_rates`."""
# Aliasing sys.modules -- rather than re-exporting names -- makes this
# module *be* the real one, so the monkey-patch idiom used by the
# notebooks (`rcs.SUBSTEPS_PER_INTERVAL = 5`) still assigns onto the
# live module rather than onto a dead copy.
import sys
import dislocluster_code.zerod.reaction_rates as _target
sys.modules[__name__] = _target
