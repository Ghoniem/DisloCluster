"""Moved to :mod:`dislocluster_code.zerod.rate_equations`."""
# Aliasing sys.modules -- rather than re-exporting names -- makes this
# module *be* the real one, so the monkey-patch idiom used by the
# notebooks (`rcs.SUBSTEPS_PER_INTERVAL = 5`) still assigns onto the
# live module rather than onto a dead copy.
import sys
import dislocluster_code.zerod.rate_equations as _target
sys.modules[__name__] = _target
