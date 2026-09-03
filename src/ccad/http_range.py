"""Minimal seekable HTTP Range reader used for revision-pinned public artifacts."""

from __future__ import annotations

import io


class RequestsRangeReader(io.RawIOBase):
    def __init__(self, session, url: str, size: int, block_size: int):
        super().__init__()
        if size <= 0 or block_size <= 0:
            raise ValueError("size and block_size must be positive")
        self.session = session
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        self.range_requests = 0
        self.bytes_received = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = min(target, self.size)
        return self.position

    def _fetch(self, position: int) -> None:
        start = (position // self.block_size) * self.block_size
        end = min(start + self.block_size, self.size) - 1
        response = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60
        )
        if response.status_code != 206:
            raise OSError(f"range request returned {response.status_code}, expected 206")
        expected = f"bytes {start}-{end}/{self.size}"
        if response.headers.get("Content-Range") != expected:
            raise OSError(
                f"unexpected Content-Range {response.headers.get('Content-Range')!r}; expected {expected!r}"
            )
        payload = response.content
        if len(payload) != end - start + 1:
            raise OSError("range response length mismatch")
        self.cache_start = start
        self.cache = payload
        self.range_requests += 1
        self.bytes_received += len(payload)

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        remaining = min(size, self.size - self.position)
        chunks: list[bytes] = []
        while remaining:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(self.position)
                cache_end = self.cache_start + len(self.cache)
            take = min(remaining, cache_end - self.position)
            offset = self.position - self.cache_start
            chunks.append(self.cache[offset:offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)
