"""Raspberry Pi IMU and foot contact sensors loaded only on hardware."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Thread
import time

import numpy as np


class ImuSensor:
    def __init__(self, upside_down: bool = False, frequency_hz: int = 50):
        try:
            import adafruit_bno055
            import board
            import busio
        except ImportError as error:
            raise RuntimeError("Install the `robot` extra on the Raspberry Pi") from error
        self.adafruit_bno055 = adafruit_bno055
        self.sensor = adafruit_bno055.BNO055_I2C(busio.I2C(board.SCL, board.SDA))
        self.sensor.mode = adafruit_bno055.NDOF_MODE
        sign_y = adafruit_bno055.AXIS_REMAP_NEGATIVE if upside_down else adafruit_bno055.AXIS_REMAP_POSITIVE
        sign_z = adafruit_bno055.AXIS_REMAP_NEGATIVE if upside_down else adafruit_bno055.AXIS_REMAP_POSITIVE
        self.sensor.axis_remap = (
            adafruit_bno055.AXIS_REMAP_Y, adafruit_bno055.AXIS_REMAP_X,
            adafruit_bno055.AXIS_REMAP_Z, adafruit_bno055.AXIS_REMAP_NEGATIVE,
            sign_y, sign_z,
        )
        self.period = 1.0 / frequency_hz
        self.queue = Queue(maxsize=1)
        self.last_update = 0.0
        self.last = None
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            started = time.monotonic()
            try:
                # BNO055 in NDOF mode fuses out a gravity vector, which is
                # what an MJLab policy's projected_gravity term wants: the
                # accelerometer alone carries the robot's own motion and is
                # only gravity while standing still. Axis remap above already
                # applies, so this is in the same frame as gyro.
                data = {
                    "gyro": np.asarray(self.sensor.gyro, dtype=float),
                    "accelerometer": np.asarray(self.sensor.acceleration, dtype=float),
                    "gravity": np.asarray(self.sensor.gravity, dtype=float),
                }
                if all(value.shape == (3,) and np.isfinite(value).all() for value in data.values()):
                    try:
                        self.queue.put_nowait(data)
                    except Full:
                        try:
                            self.queue.get_nowait()
                        except Empty:
                            pass
                        self.queue.put_nowait(data)
                    self.last_update = time.monotonic()
            except Exception:
                pass
            time.sleep(max(0.0, self.period - (time.monotonic() - started)))

    def read(self) -> dict:
        try:
            self.last = self.queue.get_nowait()
        except Empty:
            pass
        if self.last is None or time.monotonic() - self.last_update > 0.25:
            raise RuntimeError("IMU data is stale")
        return self.last


class FootContacts:
    def __init__(self):
        try:
            import board
            import digitalio
        except ImportError as error:
            raise RuntimeError("Install the `robot` extra on the Raspberry Pi") from error
        self.digitalio = digitalio
        self.pins = []
        for pin in (board.D22, board.D27):
            item = digitalio.DigitalInOut(pin)
            item.direction = digitalio.Direction.INPUT
            item.pull = digitalio.Pull.UP
            self.pins.append(item)

    def read(self):
        return np.asarray([not pin.value for pin in self.pins], dtype=float)

    def close(self):
        for pin in self.pins:
            pin.deinit()

