"""Thin HTTP helpers (stdlib only) with explicit proxy support.

``urllib`` reads ``http_proxy`` / ``https_proxy`` implicitly, which is usually
the opposite of what you want here: the upstreams are domestic Chinese
endpoints, so the default is *direct* and a proxy must be configured on
purpose.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import socket
import ssl
from http.client import HTTPConnection, HTTPSConnection
import threading
from typing import Dict, Optional, Tuple
from urllib.parse import unquote, urlparse, urlunparse

__all__ = ["request", "open_stream", "HttpResult"]

_SSL_CTX = None
_SSL_LOCK = threading.Lock()

def _ssl_context():
    global _SSL_CTX
    if _SSL_CTX is None:
        with _SSL_LOCK:
            if _SSL_CTX is None:
                _SSL_CTX = ssl.create_default_context()
    return _SSL_CTX

_POOL_LOCK = threading.Lock()
_POOL = {}
_POOL_MAX_PER_HOST = 8

def _pool_key(url, proxy):
    u = _split(url)
    return (u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), proxy or "")

def _checkout(key, timeout):
    with _POOL_LOCK:
        lst = _POOL.get(key)
        if not lst:
            return None
        conn = lst.pop()
    try:
        if getattr(conn, "sock", None) is not None:
            conn.sock.settimeout(timeout)
    except Exception:
        pass
    return conn

def _checkin(key, conn, resp_headers):
    try:
        if getattr(conn, "sock", None) is None:
            return
        if "close" in str(resp_headers.get("Connection", "")).lower():
            conn.close()
            return
        with _POOL_LOCK:
            lst = _POOL.setdefault(key, [])
            if len(lst) >= _POOL_MAX_PER_HOST:
                conn.close()
            else:
                lst.append(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

HttpResult = Tuple[int, Dict[str, str], bytes]


def _split(url: str):
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {url}")
    return u


def _proxy_url(proxy: str):
    try:
        p = urlparse(proxy)
        if (p.scheme not in ("http", "socks5", "socks5h") or not p.hostname
                or p.port == 0 or p.path not in ("", "/") or p.query or p.fragment):
            raise ValueError
        return p
    except ValueError:
        raise ValueError("invalid proxy URL; expected http:// or socks5://host:port") from None


def _proxy_headers(p):
    if p.username is None:
        return {}
    credentials = (unquote(p.username) + ":" + unquote(p.password or "")).encode("utf-8")
    return {"Proxy-Authorization": "Basic " + base64.b64encode(credentials).decode("ascii")}


def _recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise OSError("SOCKS5 proxy closed during negotiation")
        data.extend(chunk)
    return bytes(data)


def _socks5_tunnel(sock, host, port, proxy):
    user = unquote(proxy.username).encode("utf-8") if proxy.username is not None else None
    password = unquote(proxy.password or "").encode("utf-8")
    if user is not None and not (1 <= len(user) <= 255 and 1 <= len(password) <= 255):
        raise ValueError("SOCKS5 username and password must contain 1-255 UTF-8 bytes")
    methods = b"\x00\x02" if user is not None else b"\x00"
    sock.sendall(b"\x05" + bytes([len(methods)]) + methods)
    version, method = _recv_exact(sock, 2)
    if version != 5 or method not in methods:
        raise OSError("SOCKS5 proxy rejected authentication methods")
    if method == 2:
        sock.sendall(b"\x01" + bytes([len(user)]) + user + bytes([len(password)]) + password)
        if _recv_exact(sock, 2) != b"\x01\x00":
            raise OSError("SOCKS5 proxy authentication failed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        domain = host.encode("idna")
        if not 1 <= len(domain) <= 255:
            raise ValueError("SOCKS5 destination hostname must contain 1-255 IDNA bytes")
        destination = b"\x03" + bytes([len(domain)]) + domain
    else:
        destination = (b"\x01" if address.version == 4 else b"\x04") + address.packed
    sock.sendall(b"\x05\x01\x00" + destination + port.to_bytes(2, "big"))
    version, reply, reserved, kind = _recv_exact(sock, 4)
    if version != 5 or reserved != 0:
        raise OSError("invalid SOCKS5 CONNECT reply")
    if reply != 0:
        raise OSError(f"SOCKS5 CONNECT failed (reply {reply})")
    if kind == 3:
        size = _recv_exact(sock, 1)[0]
    elif kind in (1, 4):
        size = 4 if kind == 1 else 16
    else:
        raise OSError("invalid SOCKS5 bound address type")
    _recv_exact(sock, size + 2)


class _SocksConnection(HTTPConnection):
    def __init__(self, host, port, *, timeout, proxy, secure):
        self.default_port = 443 if secure else 80
        super().__init__(host, port, timeout=timeout)
        self._proxy = proxy
        self._secure = secure

    def connect(self):
        p = self._proxy
        sock = socket.create_connection((p.hostname, p.port or 1080),
                                        self.timeout, self.source_address)
        try:
            _socks5_tunnel(sock, self.host, self.port, p)
            if self._secure:
                sock = _ssl_context().wrap_socket(sock, server_hostname=self.host)
            self.sock = sock
        except BaseException:
            sock.close()
            raise


def _connect(url: str, timeout: float, proxy: str = "") -> HTTPConnection:
    """Connect directly or through an explicit HTTP/SOCKS5 proxy."""
    u = _split(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    if proxy:
        p = _proxy_url(proxy)
        if p.scheme in ("socks5", "socks5h"):
            return _SocksConnection(u.hostname, port, timeout=timeout, proxy=p,
                                    secure=u.scheme == "https")
        if u.scheme == "https":
            conn = HTTPSConnection(p.hostname, p.port or 80, timeout=timeout,
                                   context=_ssl_context())
            conn.set_tunnel(u.hostname, port, headers=_proxy_headers(p))
            return conn
        return HTTPConnection(p.hostname, p.port or 80, timeout=timeout)
    if u.scheme == "https":
        return HTTPSConnection(u.hostname, port, timeout=timeout, context=_ssl_context())
    return HTTPConnection(u.hostname, port, timeout=timeout)


def _request_args(url, proxy, headers):
    headers = dict(headers or {})
    u = _split(url)
    if proxy and _proxy_url(proxy).scheme == "http" and u.scheme == "http":
        headers.update(_proxy_headers(_proxy_url(proxy)))
        # HTTP forward proxies require absolute-form; tunnels use origin-form.
        target = urlunparse((u.scheme, u.netloc, u.path or "/", u.params, u.query, ""))
        return target, headers
    return _path_with_query(url), headers


def _path_with_query(url: str) -> str:
    u = _split(url)
    return u.path + (f"?{u.query}" if u.query else "")


def _proxy_scheme_ok(url: str, proxy: str) -> bool:  # pragma: no cover - helper
    return bool(proxy)


def request(url: str, *, method: str = "GET", headers: Optional[dict] = None,
            body: Optional[bytes] = None, timeout: float = 30,
            proxy: str = "") -> HttpResult:
    """Blocking request, reusing a pooled keep-alive connection when possible."""
    target, headers = _request_args(url, proxy, headers)
    key = _pool_key(url, proxy) if not proxy else None
    conn = _checkout(key, timeout) if key else None
    pooled = conn is not None
    if conn is None:
        conn = _connect(url, timeout, proxy)
    try:
        conn.request(method, target, body=body,
                     headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        hdrs = {k: v for k, v in resp.getheaders()}
        if key:
            _checkin(key, conn, hdrs)
        else:
            conn.close()
        return resp.status, hdrs, data
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        if pooled:
            conn = _connect(url, timeout, proxy)
            conn.request(method, target, body=body,
                         headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            hdrs = {k: v for k, v in resp.getheaders()}
            if key:
                _checkin(key, conn, hdrs)
            else:
                conn.close()
            return resp.status, hdrs, data
        raise


def open_stream(url: str, *, method: str = "POST", headers: Optional[dict] = None,
                body: Optional[bytes] = None, timeout: float = 3600,
                proxy: str = ""):
    """Open a connection without consuming the body → ``(conn, response)``.

    The caller owns both objects and must close the connection.
    """
    target, headers = _request_args(url, proxy, headers)
    conn = _connect(url, timeout, proxy)
    try:
        conn.request(method, target, body=body,
                     headers=headers or {})
        resp = conn.getresponse()
    except Exception:
        conn.close()
        raise
    return conn, resp


def json_body(obj) -> Tuple[bytes, Dict[str, str]]:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return payload, {"Content-Type": "application/json; charset=utf-8",
                     "Content-Length": str(len(payload))}
