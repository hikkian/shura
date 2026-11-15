"""Native checkpoint control must not bypass model/mode ownership via public HTTP."""
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]


class CheckpointHTTPBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="checkpoint-http-")
        root = Path(cls.tmp.name)
        for name in ("model-launch", "guardian"):
            (root / (name + ".json")).write_bytes((REPO / "config" / (name + ".example.json")).read_bytes())
        cls.modules = []
        for name in ("ai_gateway", "_gateway_legacy"):
            spec = importlib.util.spec_from_file_location("checkpoint_http_" + name, REPO / "gateway" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, AI_GATEWAY_CONFIG=str(root), AI_GATEWAY_STATE=str(root / "state")):
                spec.loader.exec_module(module)
            cls.modules.append(module)
        cls.guard = staticmethod(cls.modules[0].private_checkpoint_handler)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def handler(self, module, path, method):
        cls = self.guard(module.Handler)
        obj = cls.__new__(cls)
        obj.path, obj.command = path, method
        body = json.dumps({"model": "occamy", "filename": "tiel-coder-text.bin"}).encode()
        obj.rfile = io.BytesIO(body)
        obj.headers = {"Content-Length": str(len(body))}
        obj.send_json = Mock()
        obj.proxy = Mock()
        return obj

    def test_mutations_never_acquire_model_or_reach_backend(self):
        for module in self.modules:
            for method in ("POST", "PUT", "DELETE", "PATCH"):
                for path in ("/slots/0?action=restore", "/slots/0?action=save", "/slots/0?action=erase"):
                    with self.subTest(module=module.__name__, method=method, path=path):
                        obj = self.handler(module, path, method)
                        entry = "acquire_model" if hasattr(module, "acquire_model") else "start_llama"
                        with patch.object(module, entry) as load:
                            obj.handle_any()
                        self.assertEqual(obj.send_json.call_args.args[0], 403)
                        self.assertTrue(obj.close_connection)
                        load.assert_not_called()
                        obj.proxy.assert_not_called()

    def test_encoded_or_normalized_checkpoint_routes_are_private(self):
        for module in self.modules:
            for path in ("/slots%2F0?action=restore", "/x/../slots/0?action=restore", "//slots/0?action=restore"):
                with self.subTest(module=module.__name__, path=path):
                    obj = self.handler(module, path, "POST")
                    obj.handle_any()
                    self.assertEqual(obj.send_json.call_args.args[0], 403)
                    obj.proxy.assert_not_called()

    def test_read_and_inference_requests_delegate_unchanged(self):
        class Backend:
            def handle_any(self):
                self.called = True
                return "delegated"
        protected = self.guard(Backend)
        for path, method in (("/slots", "GET"), ("/slots/0", "HEAD"), ("/v1/chat/completions", "POST"), ("/guardian/unload", "POST")):
            with self.subTest(path=path, method=method):
                obj = protected()
                obj.path, obj.command = path, method
                self.assertEqual(obj.handle_any(), "delegated")
                self.assertTrue(obj.called)


if __name__ == "__main__":
    unittest.main()
