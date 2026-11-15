"""Who may talk to the gateway, and how a request is framed.

The gateway listens on the loopback interface, but a web page open in the user's browser can still reach it: through
`fetch`, a form post, an image tag, or DNS rebinding (an attacker's name that resolves to 127.0.0.1). Nothing here needs a
password; it only refuses what a browser can send on behalf of a stranger:

* the Host header must name this machine (rebinding sends the attacker's name);
* an Origin header must be the gateway's own, or one the user listed in `allowedOrigins`;
* a request a browser marks as cross-site (`Sec-Fetch-Site`) is refused even without an Origin (an `<img>` tag has none);
* everything under `/guardian/` that changes something must be a POST (with memoryEnabled the memory reads in GUARDIAN_READS are GETs);
* the request body is read with a length limit, so a bad `Content-Length` cannot hold memory or a thread.

`guarded_handler` wraps either gateway entry point, so the default (legacy) path and the full path share it.
Standard library only.
"""
import hmac
import io
import ipaddress
import json
import re
import socket
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}
WILDCARD_BINDS = {"0.0.0.0", "::", ""}
DEFAULTS = {
    "allowedHosts": [],                  # extra Host names (not IPs) the gateway answers to, for example a LAN name
    "allowedOrigins": [],                # web pages allowed to call the gateway from a browser, e.g. "http://localhost:3000"
    "maxRequestBytes": 32 * 1024 * 1024,  # a long chat with images is a few MB; the largest legitimate body is far below this
    "maxConnections": 64,
    "clientIdleSeconds": 300,            # an idle keep-alive connection is dropped after this long
    "requestHeaderSeconds": 30,          # from the first byte of a request to the end of its headers, in total
    "requestBodySeconds": 120,           # to receive the whole body, in total
}
SAFE_FETCH_SITES = {"same-origin", "none"}
READ_ONLY_PAGES = {"/monitor", "/monitor/"}
# LL1: the knowledge store's reads under /guardian/memory/ change nothing and are plain GETs (still behind the API key when one is set).
GUARDIAN_READS = {"/guardian/memory/search", "/guardian/memory/read", "/guardian/memory/export", "/guardian/memory/stats", "/guardian/memory/summary", "/guardian/memory/list"}


def host_name(value):
    """'Localhost:8080' -> 'localhost'; '[::1]:8080' -> '::1'; '' when it cannot be read."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        match = re.fullmatch(r"\[([0-9a-f:.]+)\](?::[0-9]{1,5})?", value)     # only an IPv6 literal can be bracketed
        return match.group(1) if match else ""
    if value.count(":") == 1:
        name, port = value.split(":")
        return name if re.fullmatch(r"[0-9]{1,5}", port) else ""
    return value


def is_ip(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def string_list(value):
    """A setting that should be a list of strings: a lone string is one entry, anything else unusable is empty."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str) and item]
    return []


def origin_of(value):
    parts = urlsplit(value or "")
    return f"{parts.scheme}://{parts.netloc}".lower() if parts.scheme and parts.netloc else ""


class Policy:
    """Decisions only; no sockets. `check` returns None to allow, or (status, code, message) to refuse."""

    def __init__(self, config, bind_host="127.0.0.1", api_key=None):
        self.config = {**DEFAULTS, **{k: config[k] for k in DEFAULTS if k in config}}
        self.bind_host = (bind_host or "").strip().lower()
        self.wildcard = self.bind_host in WILDCARD_BINDS
        self.names = set(LOOPBACK_NAMES) | {n.lower() for n in string_list(self.config["allowedHosts"])}
        self.origins = {origin_of(o) or o.lower() for o in string_list(self.config["allowedOrigins"])}
        self.api_key = api_key or None
        # LL1: the knowledge store's read endpoints are plain GETs, but only once the memory is switched on; with the flag off
        # (the default) every /guardian/ path except the status keeps requiring POST, exactly as before.
        self.guardian_reads = GUARDIAN_READS if config.get("memoryEnabled") is True else frozenset()

    # -- Host / Origin ------------------------------------------------------------------------------------
    def host_allowed(self, header):
        if header is None:
            return True                      # HTTP/1.0 clients; a browser always sends Host
        name = host_name(header)
        if not name:
            return False
        if name in self.names:
            return True
        if is_ip(name):
            address = ipaddress.ip_address(name)
            return address.is_loopback or self.wildcard or name == self.bind_host
        return False

    def origin_allowed(self, origin, host_header):
        if origin is None:
            return True
        origin = origin.strip().lower()
        if origin in self.origins or origin_of(origin) in self.origins:
            return True
        parts = urlsplit(origin)
        return parts.scheme == "http" and bool(parts.netloc) and parts.netloc == (host_header or "").strip().lower()

    def check(self, command, path, headers):
        host = headers.get("Host")
        if not self.host_allowed(host):
            return 421, "SHURA_BAD_HOST", "This gateway answers only to its own address (Host header)."
        origin = headers.get("Origin")
        if origin is not None and not self.origin_allowed(origin, host):
            return 403, "SHURA_FOREIGN_ORIGIN", "Requests from other web pages are not allowed (set allowedOrigins to permit one)."
        site = (headers.get("Sec-Fetch-Site") or "").lower()
        if origin is None and site and site not in SAFE_FETCH_SITES:
            navigate = (headers.get("Sec-Fetch-Mode") or "").lower() == "navigate"
            if not (navigate and command in ("GET", "HEAD") and path in READ_ONLY_PAGES):
                return 403, "SHURA_CROSS_SITE", "Cross-site requests are not allowed."
        if path.startswith("/guardian/") and path != "/guardian/status" and path not in self.guardian_reads and command not in ("POST", "OPTIONS"):
            return 405, "SHURA_POST_REQUIRED", "This endpoint changes state: use POST."
        if self.api_key and self.protected(command, path) and not self.key_ok(headers):
            return 401, "SHURA_KEY_REQUIRED", "A valid API key is required (Authorization: Bearer ...)."
        return None

    def protected(self, command, path):
        """With a key configured everything that changes something or wakes the model needs it; reading the status and the
        monitor's page (the monitor's data has its own key check) does not."""
        if command in ("GET", "HEAD", "OPTIONS"):
            return path.startswith("/guardian/") and path != "/guardian/status" and command != "OPTIONS"
        return not path.startswith("/monitor")

    def key_ok(self, headers):
        if hasattr(headers, "get_all"):
            values = headers.get_all("Authorization") or []
        else:
            value = headers.get("Authorization")
            values = [] if value is None else [value]
        if len(values) != 1:
            return False
        value = values[0].strip()
        if not value.lower().startswith("bearer "):
            return False
        try:
            # http.server decodes header bytes as latin-1; the key is compared as the bytes the client sent.
            return hmac.compare_digest(value[7:].strip().encode("latin-1"), str(self.api_key).encode("utf-8"))
        except (AttributeError, UnicodeError):
            return False

    def cors_origin(self, headers):
        """The Origin to echo back, only when the user listed it; the gateway never answers `*`."""
        origin = (headers.get("Origin") or "").strip().lower()
        return origin if origin and (origin in self.origins or origin_of(origin) in self.origins) else None


class BadFraming(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


HEX_SIZE = re.compile(rb"[0-9a-fA-F]{1,8}")
MAX_TRAILER_LINES = 32


def read_chunked(rfile, limit):
    """Decode a `Transfer-Encoding: chunked` request body (trailers are read and dropped), refusing more than `limit` bytes."""
    body = bytearray()
    while True:
        line = rfile.readline(1024)
        if not line.endswith(b"\n"):
            raise BadFraming(400, "Malformed chunked request body.")
        token = line.split(b";", 1)[0].rstrip(b" \t\r\n")
        if not HEX_SIZE.fullmatch(token):
            raise BadFraming(400, "Malformed chunk size.")
        size = int(token, 16)
        if size == 0:
            for _ in range(MAX_TRAILER_LINES):    # trailers, until the empty line
                trailer = rfile.readline(8192)
                if trailer in (b"\r\n", b"\n", b""):
                    return bytes(body)
            raise BadFraming(400, "Too many trailer lines.")
        if len(body) + size > limit:
            raise BadFraming(413, "Request body is too large.")
        data = rfile.read(size)
        if len(data) != size or rfile.readline(4) not in (b"\r\n", b"\n"):
            raise BadFraming(400, "Truncated chunked request body.")
        body.extend(data)


def content_length(headers, limit):
    """None when the request has no body; raises BadFraming for a value that is not a plain length within the limit."""
    values = headers.get_all("Content-Length") if hasattr(headers, "get_all") else ([headers["Content-Length"]] if "Content-Length" in headers else [])
    if not values:
        return None
    if len(values) > 1:
        raise BadFraming(400, "More than one Content-Length.")
    raw = values[0].strip()
    if len(raw) > 18 or not raw.isascii() or not raw.isdigit():
        raise BadFraming(400, "Invalid Content-Length.")
    length = int(raw)
    if length > limit:
        raise BadFraming(413, "Request body is too large.")
    return length


def guarded_handler(base, policy_factory, idle_seconds=DEFAULTS["clientIdleSeconds"]):
    """Subclass `base` (a gateway Handler) so every request passes the policy and has a safe body before `base` sees it."""

    class Guarded(base):
        timeout = idle_seconds
        _timer = None
        _cors_origin = None

        def policy(self):
            return policy_factory()

        def setup(self):
            super().setup()
            self._request_expired = threading.Event()
            self._response_started = False
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)    # headers and body are separate small writes
            except OSError:
                pass

        # -- total time limits: a client that trickles bytes cannot hold a connection (the socket timeout is per read) --------
        def arm(self, seconds):
            self.disarm()
            timer = threading.Timer(float(seconds), self.expire)
            timer.daemon = True
            timer.start()
            self._timer = timer

        def disarm(self):
            timer, self._timer = self._timer, None
            if timer is not None:
                timer.cancel()

        def expire(self):
            self._request_expired.set()
            try:
                self.connection.shutdown(socket.SHUT_RD)
            except OSError:
                pass

        def handle_one_request(self):
            self._request_expired.clear()
            self._response_started = False
            try:
                if not self.rfile.peek(1):                  # waits (up to the idle timeout) for the first byte
                    self.close_connection = True
                    return
            except (OSError, ValueError):
                self.close_connection = True
                return
            self.arm(self.policy().config["requestHeaderSeconds"])
            try:
                super().handle_one_request()
            finally:
                self.disarm()
                if self._request_expired.is_set() and not self._response_started:
                    self.refuse(408, "SHURA_REQUEST_TIMEOUT", "The request headers did not arrive in time.")

        def send_response(self, code, message=None):
            self._response_started = True
            super().send_response(code, message)

        def send_error(self, code, message=None, explain=None):
            if self._request_expired.is_set() and not self._response_started:
                return self.refuse(408, "SHURA_REQUEST_TIMEOUT", "The request headers did not arrive in time.")
            return super().send_error(code, message, explain)

        def end_headers(self):
            origin = self._cors_origin
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            super().end_headers()

        def refuse(self, status, code, message, allow=None):
            self.close_connection = True       # an unread body must not be taken for the next request
            body = json.dumps({"error": {"message": message, "type": "forbidden", "code": code}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if allow:
                self.send_header("Allow", allow)
            self.end_headers()
            if getattr(self, "command", "") != "HEAD":
                self.wfile.write(body)

        def verdict(self):
            """None to go on, or the refusal for this request (target form, Host, Origin, fetch metadata, key)."""
            target = self.path
            if not target.startswith("/") and not (self.command == "OPTIONS" and target == "*"):
                # An absolute-form target (http://host/slots/0) would slip past every path check below and still be forwarded.
                return 400, "SHURA_BAD_TARGET", "Only origin-form request targets are accepted."
            return self.policy().check(self.command, target.split("?", 1)[0], self.headers)

        def handle_expect_100(self):
            self._cors_origin = None
            verdict = self.verdict()
            if verdict:
                self.refuse(*verdict)
                return False
            return super().handle_expect_100()

        def read_request_body(self, policy):
            """Read the whole body here, under a time limit, and hand it on as a plain Content-Length body. Raises BadFraming."""
            limit = int(policy.config["maxRequestBytes"])
            encodings = self.headers.get_all("Transfer-Encoding") or []
            if encodings:
                if len(encodings) != 1 or encodings[0].strip().lower() != "chunked":
                    raise BadFraming(501, "Unsupported Transfer-Encoding.")
                if self.headers.get_all("Content-Length"):
                    raise BadFraming(400, "Transfer-Encoding together with Content-Length.")
            length = content_length(self.headers, limit)
            if not encodings and not length:
                return None
            self.arm(policy.config["requestBodySeconds"])
            try:
                if encodings:
                    body = read_chunked(self.rfile, limit)
                else:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise BadFraming(400, "Truncated request body.")
            except BadFraming:
                if self._request_expired.is_set():
                    raise BadFraming(408, "The request body did not arrive in time.") from None
                raise
            except (OSError, ValueError):
                raise BadFraming(408, "The request body did not arrive in time.") from None
            finally:
                self.disarm()
            return body

        def handle_any(self):
            self.disarm()                        # the headers are complete
            if self._request_expired.is_set():
                return self.refuse(408, "SHURA_REQUEST_TIMEOUT", "The request headers did not arrive in time.")
            self._cors_origin = None
            policy = self.policy()
            verdict = self.verdict()
            if verdict:
                return self.refuse(*verdict, allow="POST" if verdict[1] == "SHURA_POST_REQUIRED" else None)
            self._cors_origin = policy.cors_origin(self.headers)
            if self.command == "OPTIONS":
                # Preflight is answered here: it never reaches llama-server and never wakes the model.
                self.close_connection = self.close_connection or bool(self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"))
                self.send_response(204)
                self.send_header("Allow", "GET, POST, OPTIONS")
                if self._cors_origin:
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers") or "Content-Type, Authorization")
                    self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            original = self.rfile
            try:
                body = self.read_request_body(policy)
            except BadFraming as error:
                return self.refuse(error.status, "SHURA_BAD_REQUEST", error.message)
            if body is not None:
                del self.headers["Transfer-Encoding"]
                del self.headers["Content-Length"]
                self.headers["Content-Length"] = str(len(body))
                self.rfile = io.BufferedReader(io.BytesIO(body))
            try:
                return super().handle_any()
            finally:
                self.rfile = original

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_any

    Guarded.__name__ = "Guarded" + base.__name__
    return Guarded


class BoundedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on simultaneous connections: more are closed at once instead of exhausting file
    descriptors (which makes accept() fail in a tight loop)."""
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler, max_connections=DEFAULTS["maxConnections"]):
        self._slots = threading.BoundedSemaphore(max(8, int(max_connections)))
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()
