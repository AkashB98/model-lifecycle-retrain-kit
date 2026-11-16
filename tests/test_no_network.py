"""No-network enforcement: with sockets blocked, the pipeline still works."""
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Blocker:
    def __init__(self, *a, **k):
        raise RuntimeError("network access blocked in hermetic test")


class TestNoNetwork(unittest.TestCase):
    def setUp(self):
        self._real_socket = socket.socket
        socket.socket = _Blocker
        import urllib.request as urlreq
        self._real_urlopen = urlreq.urlopen
        urlreq.urlopen = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("network access blocked in hermetic test"))

    def tearDown(self):
        socket.socket = self._real_socket
        import urllib.request as urlreq
        urlreq.urlopen = self._real_urlopen

    def test_full_pipeline_offline(self):
        from generator import generate_all
        from lifecycle import run_lifecycle
        report, _ = run_lifecycle(generate_all(), scenario="drift")
        self.assertTrue(report["promoted"])

    def test_no_module_opens_sockets_at_import(self):
        for mod in ("generator", "model", "metrics", "registry", "lifecycle", "cli", "demo"):
            sys.modules.pop(mod, None)
        import cli, demo, generator, lifecycle, metrics, model, registry  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
