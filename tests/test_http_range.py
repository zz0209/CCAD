import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccad.http_range import RequestsRangeReader  # noqa: E402


class _Response:
    def __init__(self, payload: bytes, start: int, end: int, total: int):
        self.status_code = 206
        self.content = payload[start:end + 1]
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


class _Session:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.ranges = []

    def get(self, _url, *, headers, timeout):
        self.assert_timeout = timeout
        unit, spec = headers["Range"].split("=")
        assert unit == "bytes"
        start, end = (int(value) for value in spec.split("-"))
        self.ranges.append((start, end))
        return _Response(self.payload, start, end, len(self.payload))


class HTTPRangeTests(unittest.TestCase):
    def test_seek_read_and_cache(self) -> None:
        payload = bytes(range(20))
        session = _Session(payload)
        reader = RequestsRangeReader(session, "https://example.test/file", len(payload), 8)
        self.assertEqual(reader.read(3), payload[:3])
        self.assertEqual(reader.read(2), payload[3:5])
        self.assertEqual(session.ranges, [(0, 7)])
        reader.seek(-4, io.SEEK_END)
        self.assertEqual(reader.read(), payload[-4:])
        self.assertEqual(session.ranges[-1], (16, 19))
        self.assertEqual(reader.bytes_received, 12)

    def test_invalid_and_negative_seek_fail(self) -> None:
        with self.assertRaises(ValueError):
            RequestsRangeReader(_Session(b"x"), "u", 0, 1)
        reader = RequestsRangeReader(_Session(b"abc"), "u", 3, 2)
        with self.assertRaises(ValueError):
            reader.seek(-1)


if __name__ == "__main__":
    unittest.main()
