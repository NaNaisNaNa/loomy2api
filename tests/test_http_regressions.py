"""Offline regressions for persistent request bodies and safe URL diagnostics."""

import http.client
import json
import os
import secrets
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from loomy2api.config import ENV_MAP, load_config
from loomy2api.panel import Panel
from loomy2api.server import Gateway, Handler, Server


class Recorder:
    def __init__(self):
        self.lines = []

    def __call__(self, message, **kwargs):
        self.lines.append(str(message))

    def bind(self, **kwargs):
        pass

    def unbind(self, *args):
        pass


class TestHttpRegressions(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "config.json"
        self.key, self.other, self.stale, self.password = [
            secrets.token_hex(24) for _ in range(4)]
        self.path.write_text(json.dumps({
            "api_keys": [self.key], "panel_password": self.password,
            "accounts_file": str(self.path.parent / "accounts.json"),
        }), encoding="utf-8")
        env = mock.patch.dict(os.environ, {name: "" for name in ENV_MAP})
        env.start()
        self.addCleanup(env.stop)
        self.log = Recorder()
        with mock.patch("loomy2api.server.AccountPool"), \
                mock.patch("loomy2api.server.ModelGateway") as upstream, \
                mock.patch("loomy2api.server.UsageLedger"), \
                mock.patch.object(Panel, "_load_last_sms", return_value={}):
            self.gateway = Gateway(load_config(self.path), log=self.log)
        upstream.reset_mock()  # Ignore construction; fail on any later upstream use.
        self.addCleanup(lambda: self.assertTrue(not upstream.mock_calls,
                                                "Unexpected upstream use"))
        bound = type("RegressionHandler", (Handler,),
                     {"gateway": self.gateway, "_login_fails": {}})
        self.server = Server(("127.0.0.1", 0), bound)
        self.addCleanup(self.server.server_close)
        worker = threading.Thread(target=self.server.serve_forever,
                                  kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(self.server.shutdown)
        self.conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        self.addCleanup(self.conn.close)

    def request(self, method, route, payload=None, headers=None):
        body = json.dumps(payload) if payload is not None else None
        headers = {"Content-Type": "application/json", **(headers or {})}
        self.conn.request(method, route, body=body, headers=headers)
        response = self.conn.getresponse()
        return response.status, response.read().decode("utf-8")

    def login(self):
        status, body = self.request("POST", "/api/panel/login",
                                    {"password": self.password})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body).get("api_key") == self.key,
                        "Login did not return the fixture key")
        return {"Authorization": "Bearer " + self.key}

    def test_same_tcp_login_then_config_preserves_second_body(self):
        headers = self.login()
        sock = self.conn.sock
        self.assertIsNotNone(sock)
        status, _ = self.request("POST", "/api/panel/config",
                                 {"config": {"pick_top_n": 9}}, headers)
        self.assertEqual(status, 200)
        self.assertIs(self.conn.sock, sock)
        self.assertEqual(self.gateway.cfg["pick_top_n"], 9)
        self.assertEqual(json.loads(self.path.read_text())["pick_top_n"], 9)

    def test_same_tcp_post_get_post_preserves_each_body(self):
        headers = self.login()
        self.conn.close()  # Start the POST/GET/POST sequence on a fresh connection.
        status, _ = self.request("POST", "/api/panel/config",
                                 {"config": {"pick_top_n": 7}}, headers)
        self.assertEqual(status, 200)
        sock = self.conn.sock
        self.assertIsNotNone(sock)
        status, body = self.request("GET", "/api/panel/config", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["config"]["pick_top_n"], 7)
        self.assertIs(self.conn.sock, sock)
        status, _ = self.request("POST", "/api/panel/config",
                                 {"config": {"strategy": "lru"}}, headers)
        self.assertEqual(status, 200)
        self.assertIs(self.conn.sock, sock)
        saved = json.loads(self.path.read_text())
        self.assertEqual((saved["pick_top_n"], saved["strategy"]), (7, "lru"))

    def query(self, name, encoded):
        def encode(text):
            return "".join("%%%02X" % ord(c) for c in text) if encoded else text
        return ("?debug=keep&empty=&" + encode(name) + "=" + encode(self.key)
                + "&" + encode(name) + "=" + encode(self.other))

    def assert_redacted(self, body, echoed=False):
        logs = "\n".join(self.log.lines)
        for observed in (logs, body):
            leaked = any(secret in unquote(observed)
                         for secret in (self.key, self.other, self.stale))
            self.assertFalse(leaked, "A complete synthetic credential was exposed")
        self.assertTrue("debug=keep" in logs, "Safe query diagnostics were lost")
        if echoed:
            self.assertTrue("debug=keep" in body, "Safe query echo was lost")

    def test_stale_header_masks_valid_query_without_logging_credentials(self):
        for path in ("/api/panel/config", "/v1/unmapped"):
            for name in ("api_key", "apikey", "key", "access_token"):
                for encoded in (False, True):
                    with self.subTest(path=path, name=name, encoded=encoded):
                        self.conn.close()  # Error responses intentionally close the TCP connection.
                        self.log.lines.clear()
                        status, body = self.request(
                            "GET", path + self.query(name, encoded),
                            headers={"Authorization": "Bearer " + self.stale})
                        self.assertEqual(status, 401)
                        self.assertTrue(any("[auth]" in s for s in self.log.lines))
                        self.assert_redacted(body)

    def test_valid_query_unknown_routes_redact_logs_and_response_echo(self):
        for method in ("GET", "POST"):
            for name in ("api_key", "apikey", "key", "access_token"):
                for encoded in (False, True):
                    with self.subTest(method=method, name=name, encoded=encoded):
                        self.conn.close()  # Error responses intentionally close the TCP connection.
                        self.log.lines.clear()
                        status, body = self.request(
                            method, "/v1/unmapped" + self.query(name, encoded),
                            {} if method == "POST" else None)
                        self.assertEqual(status, 404)
                        self.assert_redacted(body, echoed=True)
