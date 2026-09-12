"""Low-rate, read-only telemetry for Feetech STS servos."""

from __future__ import annotations

import os
import select
import termios
import time


class RegisterReader:
    """Share rustypot's configured tty without opening/reconfiguring it again."""

    def __init__(self, port: str):
        real_port = os.path.realpath(port)
        matches = []
        for entry in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{entry}") == real_port:
                    matches.append(int(entry))
            except OSError:
                pass
        if len(matches) != 1:
            raise RuntimeError("Cannot resolve unique servo bus file descriptor")
        self.fd = os.dup(matches[0])

    def read(self, servo_id: int, address: int, length: int) -> tuple[int, bytes]:
        body = bytes([servo_id, 4, 2, address, length])
        packet = b"\xff\xff" + body + bytes([(~sum(body)) & 255])
        termios.tcflush(self.fd, termios.TCIFLUSH)
        if os.write(self.fd, packet) != len(packet):
            raise OSError("Incomplete telemetry read request")
        buf = bytearray()
        deadline = time.monotonic() + 0.008
        try:
            while time.monotonic() < deadline:
                wait = max(0.0, deadline - time.monotonic())
                if not select.select([self.fd], [], [], wait)[0]:
                    break
                try:
                    buf.extend(os.read(self.fd, 256))
                except BlockingIOError:
                    continue
                while len(buf) >= 4:
                    start = buf.find(b"\xff\xff")
                    if start < 0:
                        buf[:] = buf[-1:]
                        break
                    if start:
                        del buf[:start]
                    if len(buf) < 4:
                        break
                    size = buf[3] + 4
                    if size < 6 or size > 80:
                        del buf[0]
                        continue
                    if len(buf) < size:
                        break
                    frame = bytes(buf[:size])
                    del buf[:size]
                    if (
                        frame[2] == servo_id
                        and size == length + 6
                        and (sum(frame[2:]) & 255) == 255
                    ):
                        return frame[4], frame[5:-1]
            raise TimeoutError(f"No valid telemetry from servo {servo_id}")
        except (OSError, TimeoutError):
            # Do not leave a partial reply for rustypot's next transaction.
            termios.tcflush(self.fd, termios.TCIFLUSH)
            raise

    def close(self) -> None:
        os.close(self.fd)


def word(data: bytes, index: int) -> int:
    return data[index] | data[index + 1] << 8


def decode_status_block(data: bytes) -> dict[str, float | int]:
    """Decode STS registers 56..70; current/load units remain device raw."""
    return {
        "voltage_V": data[6] / 10.0,
        "temperature_raw": data[7],
        "load_word_raw": word(data, 4),
        "current_word_raw": word(data, 13),
    }
