# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Local test server based on http.server
"""

# From conda/tests; data/reposerver.py was refusing connections on Windows for shards tests.
from __future__ import annotations

import contextlib
import http.server
import queue
import socket
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def run_test_server(
    directory: str, finish_request_action: Callable | None = None
) -> http.server.ThreadingHTTPServer:
    """
    Run a test server on a random port. Inspect returned server to get port,
    shutdown etc.
    """

    class DualStackServer(http.server.ThreadingHTTPServer):
        daemon_threads = False  # These are per-request threads
        allow_reuse_address = True  # Good for tests
        request_queue_size = 64  # Should be more than the number of test packages

        def server_bind(self):
            # suppress exception when protocol is IPv4
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address):
            if finish_request_action:
                finish_request_action()
            self.RequestHandlerClass(request, client_address, self, directory=directory)

    def start_server(queue):
        try:
            with DualStackServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler) as httpd:
                host, port = httpd.socket.getsockname()[:2]
                queue.put(httpd)
                url_host = f"[{host}]" if ":" in host else host
                print(f"Serving HTTP on {host} port {port} (http://{url_host}:{port}/) ...")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt received, exiting.")
        except Exception as exc:
            queue.put(exc)

    started = queue.Queue()

    threading.Thread(target=start_server, args=(started,), daemon=True).start()

    result = started.get(timeout=1)
    if isinstance(result, Exception):
        raise result
    return result


if __name__ == "__main__":
    server = run_test_server(directory=".")
    print(server)
