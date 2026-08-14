"""Compatibility shim — this package moved to `dislocluster_code`.

Every former ``py_utils.X`` is now an alias for its new home, so
``from py_utils import paths``, ``python -m py_utils.X`` and the
notebooks under ``ZrMicro/code/`` keep working unchanged.
New code should import from ``dislocluster_code`` directly.
"""
