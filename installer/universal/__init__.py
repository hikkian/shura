"""Universal Shura installer core: hardware profile + model catalog -> plan.

Pure standard library. The planner in this package does no I/O: it takes a hardware profile (a dict, see
docs/HARDWARE.md) and a catalog (catalog/models.json) and returns a plan with the reasoning behind it.
"""
