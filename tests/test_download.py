import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools.audit_checkpoint import audit
from tools.download_checkpoint import fetch_range, main


class DownloadTests(unittest.TestCase):
    def test_one_failed_range_does_not_hide_a_verified_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'tree.json'
            manifest.write_text(json.dumps([
                {'path': name, 'size': 3, 'lfs': {'oid': hashlib.sha256(b'abc').hexdigest()}}
                for name in ('good', 'bad')
            ]))

            def fetch(job):
                target = job[-1]
                if target.parent.name == 'bad':
                    raise RuntimeError('network unavailable')
                target.write_bytes(b'abc')
                return 3

            argv = ['download', '--tree', str(manifest), '--destination', str(root / 'model'),
                    '--repo', 'fixture/model', '--revision', 'fixture', '--workers', '2']
            with patch('sys.argv', argv), patch('tools.download_checkpoint.fetch_range', fetch):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete checkpoint'):
                    main()
            self.assertEqual((root / 'model/good').read_bytes(), b'abc')
            self.assertFalse((root / 'model/bad').exists())

    def test_resume_preserves_range_through_permanent_redirect(self):
        data = b"model-weight-range"
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/redirect"):
                    self.send_response(308)
                    self.send_header("Location", "/blob")
                    self.end_headers()
                    return
                seen.append(self.headers["Range"])
                start, end = map(int, self.headers["Range"][6:].split("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.end_headers()
                self.wfile.write(data[start:end + 1])

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "range"
                target.with_suffix(".tmp").write_bytes(data[:3])
                fetch_range((f"http://127.0.0.1:{server.server_port}/redirect?download=true",
                             0, len(data) - 1, len(data), target))
                self.assertEqual(target.read_bytes(), data)
                self.assertEqual(seen, [f"bytes=3-{len(data)-1}"])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_audit_rejects_corruption_missing_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight").write_bytes(b"abc")
            manifest = [{"path": "weight", "size": 3,
                         "lfs": {"oid": hashlib.sha256(b"abc").hexdigest()}}]
            self.assertTrue(audit(root, manifest)[0]["verified"])
            (root / "weight").write_bytes(b"abd")
            self.assertFalse(audit(root, manifest)[0]["verified"])
            (root / "weight").unlink()
            self.assertEqual(audit(root, manifest)[0]["error"], "missing")
            manifest[0]["path"] = "../weight"
            with self.assertRaises(ValueError):
                audit(root, manifest)
