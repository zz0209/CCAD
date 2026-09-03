"""Fail-closed network guard for offline research subprocesses."""

from __future__ import annotations

import json
import os
import socket


_original_connect = socket.socket.connect


def _blocked_connect(self, address):
    log_path = os.environ.get("CCAD_SOCKET_GUARD_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"address": repr(address)}) + "\n")
    raise RuntimeError(f"CCAD offline guard blocked socket connection to {address!r}")


socket.socket.connect = _blocked_connect
