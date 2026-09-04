"""
run_all_backend_tests.py
------------------------
Executes all 6 backend test suites and aggregates results.
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

loader = unittest.TestLoader()
suite = unittest.TestSuite()

modules = [
    "test_step10_hardening",
    "test_step9_geospatial",
    "test_step8_telemetry",
    "test_step7_anomaly",
    "test_step6_fusion",
    "test_controller",
]

for mod_name in modules:
    mod = __import__(mod_name)
    suite.addTests(loader.loadTestsFromModule(mod))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print(f"\n==========================================")
print(f"TOTAL TESTS RUN: {result.testsRun}")
print(f"ERRORS: {len(result.errors)}")
print(f"FAILURES: {len(result.failures)}")
print(f"WAS SUCCESSFUL: {result.wasSuccessful()}")
print(f"==========================================")

sys.exit(0 if result.wasSuccessful() else 1)
