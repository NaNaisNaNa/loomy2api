"""Offline wire regressions: explicit proxies, TLS verification and live reads."""

import base64
from contextlib import contextmanager
import ipaddress
from pathlib import Path
import socket
import ssl
import threading
import unittest
from unittest import mock
from urllib.parse import urlparse

from loomy2api import httpc

# Public test-only certificate/key; never use this fixture outside these tests.
CERT = Path(__file__).with_name("fixtures") / "proxy-localhost.pem"


def read_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("test peer closed")
        data += chunk
    return data


def read_http(sock):
    data = b""
    while not data.endswith(b"\r\n\r\n"):
        data += read_exact(sock, 1)
    lines = data.decode("ascii").split("\r\n")
    return lines[0], dict(line.split(": ", 1) for line in lines[1:] if line)


def socks_accept(sock, record, credentials=None, bound=None):
    version, count = read_exact(sock, 2)
    if version != 5:
        raise AssertionError("wrong SOCKS version")
    record["methods"] = read_exact(sock, count)
    sock.sendall(b"\x05\x02" if credentials else b"\x05\x00")
    if credentials:
        version, size = read_exact(sock, 2)
        user = read_exact(sock, size)
        password = read_exact(sock, read_exact(sock, 1)[0])
        if version != 1 or (user, password) != credentials:
            raise AssertionError("incorrect test credentials")
        sock.sendall(b"\x01\x00")
    version, command, reserved, kind = read_exact(sock, 4)
    if (version, command, reserved) != (5, 1, 0):
        raise AssertionError("wrong CONNECT command")
    if kind == 3:
        host = read_exact(sock, read_exact(sock, 1)[0]).decode("ascii")
    else:
        host = str(ipaddress.ip_address(read_exact(sock, 4 if kind == 1 else 16)))
    record["target"] = (kind, host, int.from_bytes(read_exact(sock, 2), "big"))
    reply = b"\x05\x00\x00" + (bound or b"\x01" + b"\x00" * 6)
    for value in reply:
        sock.sendall(bytes([value]))


class TestHttpProxy(unittest.TestCase):
    @contextmanager
    def peer(self, action):
        errors = []
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)
        port = listener.getsockname()[1]

        def run():
            try:
                with listener.accept()[0] as sock:
                    sock.settimeout(3)
                    action(sock)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            yield "127.0.0.1:%d" % port
        finally:
            listener.close()
            worker.join(4)
            self.assertFalse(worker.is_alive(), "proxy fixture did not finish")
            if errors:
                raise errors[0]

    def tls_context(self):
        return ssl.create_default_context(cafile=str(CERT))

    def tls_server(self, sock):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(CERT))
        return context.wrap_socket(sock, server_side=True)

    def respond(self, sock, record):
        record["request"] = read_http(sock)
        sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")

    def test_socks_domain_resolved_by_proxy_and_origin_form(self):
        record = {}

        def serve(sock):
            socks_accept(sock, record)
            self.respond(sock, record)

        with self.peer(serve) as address:
            with mock.patch.object(socket, "create_connection", wraps=socket.create_connection) as dial:
                status, _, body = httpc.request("http://only-proxy.invalid:8123/test?q=1",
                                                proxy="socks5://" + address, timeout=2)
            self.assertEqual(dial.call_count, 1)
            self.assertEqual(dial.call_args.args[0], ("127.0.0.1", int(address.split(":")[1])))
        self.assertEqual((status, body), (200, b"ok"))
        self.assertEqual(record["target"], (3, "only-proxy.invalid", 8123))
        self.assertEqual(record["request"][0], "GET /test?q=1 HTTP/1.1")

    def test_direct_requests_reuse_connection_and_ignore_environment_proxies(self):
        record = []

        def serve(sock):
            for _ in range(2):
                record.append(read_http(sock)[0])
                sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

        with self.peer(serve) as address:
            url = "http://" + address + "/direct"
            headers = {"X-Fixture": "original"}
            with mock.patch.dict("os.environ", {"http_proxy": "http://dead.invalid:1"}):
                with mock.patch.object(socket, "create_connection", wraps=socket.create_connection) as dial:
                    for _ in range(2):
                        self.assertEqual(httpc.request(url, headers=headers, timeout=2)[2], b"ok")
                    self.assertEqual(dial.call_count, 1)
            with httpc._POOL_LOCK:
                for conn in httpc._POOL.pop(httpc._pool_key(url, ""), []):
                    conn.close()
            self.assertEqual(headers, {"X-Fixture": "original"})
        self.assertEqual(record, ["GET /direct HTTP/1.1"] * 2)

    def test_socks_url_credentials_and_socks5h_alias(self):
        for scheme in ("socks5", "socks5h"):
            record = {}

            def serve(sock):
                socks_accept(sock, record, credentials=(b"u@ser", "p:密".encode()))
                self.respond(sock, record)

            with self.subTest(scheme=scheme), self.peer(serve) as address:
                result = httpc.request("http://origin.invalid/", timeout=2,
                                      proxy=scheme + "://u%40ser:p%3A%E5%AF%86@" + address)
                self.assertEqual(result[0], 200)
            self.assertIn(2, record["methods"])

    def test_socks_literal_addresses_and_all_bound_address_types(self):
        for host, kind, bound in (("127.0.0.2", 1, b"\x03\x01x\x00\x50"),
                                   ("[::1]", 4, b"\x04" + b"\x00" * 18)):
            record = {}

            def serve(sock):
                socks_accept(sock, record, bound=bound)
                self.respond(sock, record)

            with self.subTest(host=host), self.peer(serve) as address:
                self.assertEqual(httpc.request("http://" + host + ":99/",
                                               proxy="socks5://" + address, timeout=2)[0], 200)
            self.assertEqual(record["target"], (kind, host.strip("[]"), 99))

    def test_http_forward_proxy_absolute_form_and_auth(self):
        record = {}
        with self.peer(lambda sock: self.respond(sock, record)) as address:
            self.assertEqual(httpc.request("http://origin.invalid:8123/test?q=1#fragment",
                                           proxy="http://user:p%40ss@" + address, timeout=2)[0], 200)
        line, headers = record["request"]
        self.assertEqual(line, "GET http://origin.invalid:8123/test?q=1 HTTP/1.1")
        self.assertEqual(headers["Host"], "origin.invalid:8123")
        self.assertEqual(headers["Proxy-Authorization"], "Basic " + base64.b64encode(b"user:p@ss").decode())

    def test_https_through_each_proxy_verifies_upstream_tls_and_isolates_proxy_auth(self):
        for scheme in ("socks5", "http"):
            record = {}

            def serve(sock):
                if scheme == "socks5":
                    socks_accept(sock, record)
                else:
                    record["connect"] = read_http(sock)
                    sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                with self.tls_server(sock) as tls:
                    self.respond(tls, record)

            with self.subTest(scheme=scheme), self.peer(serve) as address:
                with mock.patch.object(httpc, "_ssl_context", side_effect=self.tls_context):
                    self.assertEqual(httpc.request("https://localhost:8443/test", timeout=2,
                                                   proxy=scheme + "://user:pass@" + address)[2], b"ok")
            self.assertNotIn("Proxy-Authorization", record["request"][1])
            self.assertEqual(record["request"][0], "GET /test HTTP/1.1")
            if scheme == "http":
                self.assertTrue(record["connect"][0].startswith("CONNECT localhost:8443 HTTP/1."))
                self.assertIn("Proxy-Authorization", record["connect"][1])

    def test_tls_rejects_wrong_hostname_and_untrusted_certificate(self):
        for scheme in ("socks5", "http"):
            for host, trusted in (("wrong.invalid", True), ("localhost", False)):
                def serve(sock):
                    if scheme == "socks5":
                        socks_accept(sock, {})
                    else:
                        read_http(sock)
                        sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    with self.assertRaises(OSError):
                        self.tls_server(sock)

                context = self.tls_context() if trusted else ssl.create_default_context()
                with self.subTest(scheme=scheme, host=host), self.peer(serve) as address:
                    with mock.patch.object(httpc, "_ssl_context", return_value=context):
                        with self.assertRaises(ssl.SSLCertVerificationError):
                            httpc.request("https://" + host + "/", proxy=scheme + "://" + address, timeout=2)

    def test_stream_reads_first_event_before_generation_finishes(self):
        for scheme in ("socks5", "http"):
            for secure in (False, True):
                release, finished = threading.Event(), threading.Event()
                record = {}

                def respond(sock):
                    record["request"] = read_http(sock)
                    sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                 b"Content-Length: 27\r\nConnection: close\r\n\r\ndata: first\n\n")
                    if not release.wait(3):
                        raise AssertionError("stream consumer did not receive first event")
                    sock.sendall(b"data: second\n\n")
                    finished.set()

                def serve(sock):
                    if scheme == "socks5":
                        socks_accept(sock, record)
                    elif secure:
                        read_http(sock)
                        sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    if secure:
                        with self.tls_server(sock) as tls:
                            respond(tls)
                    else:
                        respond(sock)

                with self.subTest(scheme=scheme, secure=secure), self.peer(serve) as address:
                    with mock.patch.object(httpc, "_ssl_context", side_effect=self.tls_context):
                        conn, response = httpc.open_stream(
                            ("https://localhost" if secure else "http://only-proxy.invalid") + "/events",
                            proxy=scheme + "://" + address, timeout=2)
                    try:
                        self.assertEqual(response.readline(), b"data: first\n")
                        self.assertEqual(response.readline(), b"\n")
                        self.assertFalse(finished.is_set())
                        release.set()
                        self.assertEqual(response.read(), b"data: second\n\n")
                    finally:
                        release.set()
                        response.close()
                        conn.close()

    def test_exact_reads_handle_fragmentation_and_truncated_replies(self):
        sock = mock.Mock()
        sock.recv.side_effect = [b"\x05", b"\x00"]
        self.assertEqual(httpc._recv_exact(sock, 2), b"\x05\x00")
        self.assertEqual(sock.recv.call_args_list, [mock.call(2), mock.call(1)])
        sock.recv.side_effect = [b"\x05", b""]
        with self.assertRaisesRegex(OSError, "closed"):
            httpc._recv_exact(sock, 2)

    def test_negotiation_failures_close_socket_without_direct_fallback(self):
        replies = [b"\x05\xff", b"\x04\x00", b"\x05\x02", b"\x05",
                   b"\x05\x00\x05\x05\x00\x01",
                   b"\x05\x00\x04\x00\x00\x01",
                   b"\x05\x00\x05\x00\x01\x01",
                   b"\x05\x00\x05\x00\x00\x02",
                   b"\x05\x00\x05\x00\x00\x01\x00"]
        for reply in replies:
            pending = bytearray(reply)
            sock = mock.Mock()

            def recv(size):
                result = bytes(pending[:size])
                del pending[:size]
                return result

            sock.recv.side_effect = recv
            with self.subTest(reply=reply), mock.patch.object(socket, "create_connection", return_value=sock) as dial:
                with self.assertRaises(OSError):
                    httpc.request("http://origin.invalid/", proxy="socks5://proxy.invalid:1081", timeout=2)
                dial.assert_called_once_with(("proxy.invalid", 1081), 2, None)
                sock.close.assert_called_once()

    def test_auth_failure_and_timeout_close_socket(self):
        for result, expected in (([b"\x05\x02", b"\x01\x01"], OSError),
                                  (socket.timeout("fixture timeout"), socket.timeout)):
            sock = mock.Mock()
            sock.recv.side_effect = result
            with self.subTest(result=result), mock.patch.object(socket, "create_connection", return_value=sock):
                with self.assertRaises(expected):
                    httpc.open_stream("https://origin.invalid/", proxy="socks5://user:pass@proxy.invalid", timeout=0.1)
                sock.close.assert_called_once()

    def test_invalid_proxy_errors_do_not_echo_credentials(self):
        for value in ("https://user:private@host:99", "socks4://user:private@host",
                      "socks5://user:private@host:99999", "socks5://user:private@host:0",
                      "socks5://user:private@", "http://user:private@host/path"):
            with self.subTest(value=value), mock.patch.object(socket, "create_connection") as dial:
                with self.assertRaises(ValueError) as error:
                    httpc.request("https://origin.invalid/", proxy=value)
                self.assertNotIn("private", str(error.exception))
                dial.assert_not_called()

    def test_auth_length_validation_closes_socket_without_leaking_secret(self):
        for credentials in ("u:" + "x" * 256, "u:", ":private"):
            sock = mock.Mock()
            with self.subTest(length=len(credentials)), mock.patch.object(socket, "create_connection", return_value=sock):
                with self.assertRaises(ValueError) as error:
                    httpc.request("http://origin.invalid/", proxy="socks5://" + credentials + "@proxy.invalid")
                self.assertNotIn("private", str(error.exception))
                sock.close.assert_called_once()
                sock.sendall.assert_not_called()


if __name__ == "__main__":
    unittest.main()
