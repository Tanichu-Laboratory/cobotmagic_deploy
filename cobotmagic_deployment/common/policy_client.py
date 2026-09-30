"""Non-blocking ZeroMQ policy client used for asynchronous inference.

At most one request is in flight. ``poll()`` never blocks, so the caller
keeps executing actions while the server is inferring; after
``response_timeout_sec`` the socket is recreated (a REQ socket cannot send
again until it has received a reply) and the request is dropped.
"""

import time

import zmq

from cobotmagic_deployment.common.bridge_log import BridgeLog
from cobotmagic_deployment.common.policy_server_protocol import client_socket_kind


class AsyncPolicyClient:
    def __init__(self, connect_addr, socket_type='req', io_timeout_ms=50,
                 response_timeout_sec=5.0, log=None, context=None):
        self.connect_addr = connect_addr
        self.socket_type = socket_type
        self.io_timeout_ms = io_timeout_ms
        self.response_timeout_sec = response_timeout_sec
        self.log = log or BridgeLog()
        self.context = context or zmq.Context.instance()
        self.pending = None
        self._last_wait_warn = 0.0
        self.sock = self._make_socket()

    @classmethod
    def from_config(cls, cfg, rate_hz, log=None):
        zmq_cfg = cfg['zmq']
        return cls(
            zmq_cfg.get('client_connect', 'tcp://127.0.0.1:5557'),
            zmq_cfg.get('socket_type', 'req'),
            io_timeout_ms=max(int(1000 / rate_hz), 1),
            response_timeout_sec=float(cfg['ros'].get('policy_response_timeout_sec', 5.0)),
            log=log,
        )

    def _make_socket(self):
        sock = self.context.socket(client_socket_kind(self.socket_type))
        sock.connect(self.connect_addr)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.io_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.io_timeout_ms)
        return sock

    @property
    def busy(self):
        return self.pending is not None

    def send(self, frames):
        """Send a multipart request. Returns ``False`` on a send timeout."""
        try:
            self.sock.send_multipart(frames)
        except zmq.error.Again:
            self.log.warn("ZeroMQ send timeout; will retry next cycle.")
            return False
        return True

    def mark_pending(self, request):
        """Remember the in-flight request; ``request['time']`` is its send time."""
        self.pending = request

    def poll(self):
        """Return ``(frames, request)`` when a reply arrived, else ``None``."""
        if self.pending is None:
            return None
        try:
            frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            frames = None
            now = time.monotonic()
            elapsed = now - self.pending['time']
            if elapsed >= 2.0 and now - self._last_wait_warn >= 2.0:
                self.log.warn(
                    f"Waiting for policy server response from {self.connect_addr}. "
                    "Continuing with available chunk actions."
                )
                self._last_wait_warn = now
            if elapsed >= max(self.response_timeout_sec, 0.1):
                self.log.warn(
                    f"Policy server response timed out after {self.response_timeout_sec:.1f}s; "
                    "recreating ZeroMQ socket and keeping any available chunk actions."
                )
                self.sock.close(0)
                self.sock = self._make_socket()
                self.pending = None
        if frames:
            request = self.pending
            self.pending = None
            return frames, request
        return None

    def close(self):
        self.sock.close(0)
