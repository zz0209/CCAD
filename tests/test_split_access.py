import unittest

from ccad.split_access import SplitAccess


class SplitAccessTests(unittest.TestCase):
    def test_discovery_stop_never_calls_calibration_or_audit_loader(self):
        calls = []
        def loader(split):
            calls.append(split)
            if split != "discovery":
                raise AssertionError("forbidden early load")
            return object()
        data = SplitAccess(("discovery", "calibration"), loader)
        list(data)  # inspecting available keys must not materialize arrays
        self.assertEqual(calls, [])
        first = data["discovery"]
        self.assertIs(first, data["discovery"])
        self.assertEqual(calls, ["discovery"])
        self.assertFalse(data.requested("calibration"))
        with self.assertRaises(KeyError):
            data["audit"]
        self.assertEqual(calls, ["discovery"])

    def test_partial_loader_failure_is_still_a_read_attempt(self):
        def loader(split):
            raise OSError("partial read")
        data = SplitAccess(("calibration",), loader)
        with self.assertRaises(OSError):
            data["calibration"]
        self.assertTrue(data.requested("calibration"))
        self.assertEqual(data.events[0]["status"], "failed")
