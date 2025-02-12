import collections
import dataclasses
import datetime
import pathlib
import struct
import sys
import threading
import time
import types
import typing

import numpy
import numpy.typing
import PySide6.QtCore
import PySide6.QtGui
import PySide6.QtQml
import PySide6.QtQuick
import pypylon.pylon as pylon

dirname = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass
class Output:
    path: pathlib.Path
    width: int
    height: int
    exposure: float
    gain: float


def size_to_string(size: int) -> str:
    if size < 1000000:
        return f"{size / 1e3:.2f} kB"
    if size < 1000000000:
        return f"{size / 1e6:.2f} MB"
    if size < 1000000000000:
        return f"{size / 1e9:.2f} GB"
    return f"{size / 1e12:.2f} TB"


def milliseconds_to_string(milliseconds: int) -> str:
    hours = milliseconds // 3600000
    milliseconds -= hours * 3600000
    minutes = milliseconds // 60000
    milliseconds -= minutes * 60000
    seconds = milliseconds // 1000
    milliseconds -= seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class CircularBuffer:
    def __init__(self, length: int):
        self.data: list[typing.Optional[tuple[int, int, int, bytearray]]] = [
            None
        ] * length
        self.write_index = 0
        self.packets = 0

    def resize(self, new_length: int):
        length = len(self.data)
        if length == new_length:
            return
        new_data: list[typing.Optional[tuple[int, int, int, bytearray]]] = [
            None
        ] * new_length
        if new_length > length:
            index = self.write_index
            for new_index in range(0, length):
                new_data[new_index] = self.data[index]
                index = (index + 1) % length
                if index == self.write_index:
                    break
            self.data = new_data
            self.write_index = length
            self.packets = length
        else:
            index = (self.write_index + (length - new_length)) % length
            for new_index in range(0, new_length):
                new_data[new_index] = self.data[index]
                index = (index + 1) % length
            assert index == self.write_index
            self.data = new_data
            self.write_index = 0
            self.packets = min(self.packets, new_length)

    def clear(self):
        self.data = [None] * len(self.data)

    def push(self, timestamp: int, width: int, height: int, data: bytearray):
        self.data[self.write_index] = (timestamp, width, height, data)
        self.write_index = (self.write_index + 1) % len(self.data)
        self.packets = min(self.packets + 1, len(self.data))


class Recorder:
    def __init__(self, configuration: PySide6.QtQml.QQmlPropertyMap):
        self.mode = 0
        self.configuration = configuration
        self.queue: collections.deque[tuple[int, int, int, bytearray]] = (
            collections.deque()
        )
        self.circular_buffers: tuple[CircularBuffer, CircularBuffer] = (
            CircularBuffer(1),
            CircularBuffer(1),
        )
        self.circular_buffers_lock = threading.Lock()
        self.active_circular_buffer = 0
        self.output: typing.Optional[Output] = None
        self.running = True
        self.thread = threading.Thread(target=self.target, daemon=True)
        self.thread.start()

    def direct_mode(self):
        self.mode = 0
        with self.circular_buffers_lock:
            self.circular_buffers[0].resize(1)
            self.circular_buffers[0].resize(1)

    def circular_buffer(self, length: int):
        with self.circular_buffers_lock:
            self.circular_buffers[0].resize(length)
            self.circular_buffers[1].resize(length)
        self.mode = 1
        self.queue.clear()

    def target(self):
        current_mode = 0
        queue_output = None
        queue_output_file: typing.Optional[typing.IO[bytes]] = None
        queue_recorded_bytes = 0
        queue_recorded_frames = 0
        queue_recorded_first_timestamp = None
        last_update = 0
        while self.running:
            if queue_output is None:
                current_mode = self.mode
            new_output = self.output
            if current_mode == 0:  # direct mode
                if new_output != queue_output:
                    if queue_output_file is not None:
                        buffered_frames = len(self.queue)
                        for _ in range(0, buffered_frames):
                            try:
                                timestamp, width, height, data = self.queue.popleft()
                                queue_output_file.write(struct.pack("<Q", timestamp))  # type: ignore
                                queue_output_file.write(data)  # type: ignore
                            except IndexError:
                                break
                        queue_output_file.close()
                        queue_output_file = None
                        queue_recorded_bytes = 0
                        queue_recorded_frames = 0
                        queue_recorded_first_timestamp = None
                    queue_output = new_output
                    if queue_output is not None:
                        queue_output_file = open(queue_output.path, "wb")
                        queue_output_file.write(b"BASLER")
                        queue_output_file.write(
                            struct.pack(
                                "<HHdd",
                                queue_output.width,
                                queue_output.height,
                                queue_output.exposure,
                                queue_output.gain,
                            )
                        )
                        queue_recorded_bytes = 26
                    last_update = time.monotonic_ns()
                    self.configuration.insert(
                        "recording_bytes", f"{queue_recorded_bytes} B"
                    )
                    self.configuration.insert("recording_frames", queue_recorded_frames)
                    self.configuration.insert("recording_duration", "00:00:00.000")
                try:
                    timestamp, width, height, data = self.queue.popleft()
                    buffered_frames = len(self.queue)

                    if buffered_frames < 10:
                        buffered_frames_string = "< 10 frames"
                    else:
                        buffered_frames_string = f"{buffered_frames} frames"
                    if queue_output is None:
                        now = time.monotonic_ns()
                        if now - last_update > 10000000:  # 10 ms
                            last_update = now
                            self.configuration.insert(
                                "buffered_frames", buffered_frames_string
                            )
                    else:
                        if queue_recorded_first_timestamp is None:
                            queue_recorded_first_timestamp = timestamp
                        assert (
                            width == queue_output.width
                        ), f"{width=} {queue_output.width=}"
                        assert (
                            height == queue_output.height
                        ), f"{height=} {queue_output.height=}"
                        queue_output_file.write(struct.pack("<Q", timestamp))  # type: ignore
                        queue_output_file.write(data)  # type: ignore
                        queue_recorded_bytes += 8 + len(data)
                        queue_recorded_frames += 1
                        now = time.monotonic_ns()
                        if now - last_update > 10000000:  # 10 ms
                            last_update = now
                            self.configuration.insert(
                                "buffered_frames", buffered_frames_string
                            )
                            self.configuration.insert(
                                "recording_bytes", size_to_string(queue_recorded_bytes)
                            )
                            self.configuration.insert(
                                "recording_frames", queue_recorded_frames
                            )
                            self.configuration.insert(
                                "recording_duration",
                                milliseconds_to_string(
                                    int(
                                        round(
                                            timestamp / 1e6
                                            - queue_recorded_first_timestamp / 1e6
                                        )
                                    )
                                ),
                            )

                except IndexError:
                    now = time.monotonic_ns()
                    if now - last_update > 10000000:  # 10 ms
                        last_update = now
                        self.configuration.insert("buffered_frames", "< 10 frames")
                    time.sleep(0.005)
            else:  # circular buffer
                if new_output is None:
                    now = time.monotonic_ns()
                    if now - last_update > 10000000:  # 10 ms
                        last_update = now
                        with self.circular_buffers_lock:
                            circular_buffer_length = len(
                                self.circular_buffers[self.active_circular_buffer].data
                            )
                            circular_buffer_packets = self.circular_buffers[
                                self.active_circular_buffer
                            ].packets
                        self.configuration.insert(
                            "circular_buffer_usage",
                            f"{circular_buffer_packets} / {circular_buffer_length}",
                        )
                    time.sleep(0.005)
                else:
                    with self.circular_buffers_lock:
                        circular_buffer_length = len(
                            self.circular_buffers[self.active_circular_buffer].data
                        )
                        save_circular_buffer = self.active_circular_buffer
                        self.active_circular_buffer = (
                            self.active_circular_buffer + 1
                        ) % 2
                    with open(new_output.path, "wb") as circular_buffer_output_file:
                        circular_buffer_output_file.write(b"BASLER")
                        circular_buffer_output_file.write(
                            struct.pack(
                                "<HHdd",
                                new_output.width,
                                new_output.height,
                                new_output.exposure,
                                new_output.gain,
                            )
                        )
                        circular_buffer_bytes = 26
                        circular_buffer_frames = 0
                        circular_buffer_first_timestamp: typing.Optional[int] = None
                        write_index = self.circular_buffers[
                            save_circular_buffer
                        ].write_index
                        index = write_index
                        while True:
                            packet = self.circular_buffers[save_circular_buffer].data[
                                index
                            ]
                            if packet is not None:
                                timestamp, width, height, data = packet
                                if circular_buffer_first_timestamp is None:
                                    circular_buffer_first_timestamp = timestamp
                                circular_buffer_output_file.write(
                                    struct.pack("<Q", timestamp)
                                )
                                circular_buffer_output_file.write(data)
                                circular_buffer_bytes += 8 + len(data)
                                circular_buffer_frames += 1
                                now = time.monotonic_ns()
                                if now - last_update > 10000000:  # 10 ms
                                    last_update = now
                                    self.configuration.insert(
                                        "recording_bytes",
                                        size_to_string(circular_buffer_bytes),
                                    )
                                    self.configuration.insert(
                                        "recording_frames",
                                        circular_buffer_first_timestamp,
                                    )
                                    self.configuration.insert(
                                        "recording_duration",
                                        milliseconds_to_string(
                                            int(
                                                round(
                                                    timestamp / 1e6
                                                    - circular_buffer_first_timestamp
                                                    / 1e6
                                                )
                                            )
                                        ),
                                    )
                            self.circular_buffers[save_circular_buffer].data[
                                index
                            ] = None
                            index = (index + 1) % len(
                                self.circular_buffers[save_circular_buffer].data
                            )
                            if index == write_index:
                                break
                    self.configuration.insert("recording_bytes", "0 B")
                    self.configuration.insert("recording_frames", 0)
                    self.configuration.insert("recording_duration", "00:00:00.000")
                    self.configuration.insert("recording_name", None)
                    self.configuration.insert(
                        "circular_buffer_usage",
                        f"0 / {circular_buffer_length}",
                    )
                    self.output = None

        if queue_output_file is not None:
            buffered_frames = len(self.queue)
            for _ in range(0, buffered_frames):
                try:
                    timestamp, width, height, data = self.queue.popleft()
                    output_file.write(struct.pack("<Q", timestamp))  # type: ignore
                    output_file.write(data)  # type: ignore
                except IndexError:
                    break
            queue_output_file.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool:
        self.running = False
        self.thread.join()
        return False

    def start(
        self,
        path: pathlib.Path,
        width: int,
        height: int,
        exposure: float,
        gain: float,
    ):
        self.output = Output(
            path=path,
            width=width,
            height=height,
            exposure=exposure,
            gain=gain,
        )

    def stop(self):
        self.output = None

    def push(self, timestamp: int, width: int, height: int, data: bytearray):
        if self.mode == 0:  # direct
            self.queue.append((timestamp, width, height, data))
        else:  # circular buffer
            with self.circular_buffers_lock:
                self.circular_buffers[self.active_circular_buffer].push(
                    timestamp, width, height, data
                )


class ImageProvider(PySide6.QtQuick.QQuickImageProvider):
    def __init__(self, configuration: PySide6.QtQml.QQmlPropertyMap):
        super().__init__(PySide6.QtQml.QQmlImageProviderBase.ImageType.Image)
        self.configuration = configuration

        self.lock = threading.Lock()

        self.request_width = 0
        self.request_height = 0
        self.request_array_index = 0
        self.request_array: bytearray = bytearray([])
        self.request_timestamps = numpy.zeros(32, dtype=numpy.uint64)

        self.latest_width = 0
        self.latest_height = 0
        self.latest_array: bytearray = bytearray([])
        self.latest_array_index = 0
        self.latest_timestamps = numpy.zeros(32, dtype=numpy.uint64)
        self.latest_timestamps_index = 0

        self.unpack_buffer = numpy.zeros(0, dtype=numpy.uint16)
        self.unpacked_array = numpy.zeros(0, dtype=numpy.uint16)

    def update_array(self, timestamp: int, width: int, height: int, data: bytearray):
        with self.lock:
            self.latest_timestamps[self.latest_timestamps_index] = timestamp
            self.latest_timestamps_index = (self.latest_timestamps_index + 1) % len(
                self.latest_timestamps
            )
            self.latest_width = width
            self.latest_height = height
            self.latest_array = data
            self.latest_array_index += 1

    def _unpack_bytes(
        self,
        byte_0_offset: int,
        byte_0_mask: int,
        byte_0_shift: int,
        byte_1_offset: int,
        byte_1_mask: int,
        byte_1_shift: int,
        output_offset: int,
    ):
        if byte_0_mask == 0xFF:
            self.unpacked_array[output_offset::4] = self.request_array[byte_0_offset::5]
        else:
            numpy.bitwise_and(
                self.request_array[byte_0_offset::5],
                byte_0_mask,
                out=self.unpacked_array[output_offset::4],
                dtype=numpy.uint16,
            )
        if byte_0_shift != 0:
            if byte_0_shift > 0:
                numpy.bitwise_left_shift(
                    self.unpacked_array[output_offset::4],
                    byte_0_shift,
                    out=self.unpacked_array[output_offset::4],
                    dtype=numpy.uint16,
                )
            else:
                numpy.bitwise_right_shift(
                    self.unpacked_array[output_offset::4],
                    -byte_0_shift,
                    out=self.unpacked_array[output_offset::4],
                    dtype=numpy.uint16,
                )
        if byte_1_mask == 0xFF:
            self.unpack_buffer[::] = self.request_array[byte_1_offset::5]
        else:
            numpy.bitwise_and(
                self.request_array[byte_1_offset::5],
                byte_1_mask,
                out=self.unpack_buffer,
                dtype=numpy.uint16,
            )
        if byte_1_shift != 0:
            if byte_1_shift > 0:
                numpy.bitwise_left_shift(
                    self.unpack_buffer,
                    byte_1_shift,
                    out=self.unpack_buffer,
                    dtype=numpy.uint16,
                )
            else:
                numpy.bitwise_right_shift(
                    self.unpack_buffer,
                    -byte_1_shift,
                    out=self.unpack_buffer,
                    dtype=numpy.uint16,
                )
        numpy.bitwise_or(
            self.unpacked_array[output_offset::4],
            self.unpack_buffer,
            out=self.unpacked_array[output_offset::4],
            dtype=numpy.uint16,
        )

    def _unpack(self):
        if len(self.unpacked_array) != self.request_width * self.request_height:
            self.unpacked_array = numpy.zeros(
                self.request_width * self.request_height, dtype=numpy.uint16
            )
            self.unpack_buffer = numpy.zeros(
                self.request_width * self.request_height // 4, dtype=numpy.uint16
            )
        self._unpack_bytes(0, 0b11111111, 0, 1, 0b11, 8, 0)
        self._unpack_bytes(1, 0b11111111, -2, 2, 0b1111, 6, 1)
        self._unpack_bytes(2, 0b11111111, -4, 3, 0b111111, 4, 2)
        self._unpack_bytes(3, 0b11111111, -6, 4, 0b11111111, 2, 3)
        numpy.bitwise_left_shift(self.unpacked_array, 6, out=self.unpacked_array)

    def requestImage(
        self,
        id: str,
        size: PySide6.QtCore.QSize,
        requestedSize: PySide6.QtCore.QSize,
    ) -> PySide6.QtGui.QImage:
        with self.lock:
            self.request_width = self.latest_width
            self.request_height = self.latest_height
            self.request_array = self.latest_array
            self.request_array_index = self.latest_array_index
            self.request_timestamps[:] = self.latest_timestamps

        self.request_timestamps.sort()
        nonzero_timestamps = self.request_timestamps[self.request_timestamps > 0]
        if len(nonzero_timestamps) > 1:
            mean_timestamp_delta = numpy.mean(
                numpy.diff(self.request_timestamps[self.request_timestamps > 0])
            )
            framerate = float(1e9 / mean_timestamp_delta)
            self.configuration.insert("measured_framerate", framerate)

        if len(self.request_array) == 0:
            size.setWidth(640)
            size.setHeight(480)
            return PySide6.QtGui.QImage(
                640,
                480,
                PySide6.QtGui.QImage.Format.Format_Grayscale16,
            )
        self._unpack()
        size.setWidth(self.request_width)
        size.setHeight(self.request_height)
        image = PySide6.QtGui.QImage(
            self.unpacked_array.tobytes(),
            self.request_width,
            self.request_height,
            PySide6.QtGui.QImage.Format.Format_Grayscale16,
        )
        return image


class ImageEventHandler(pylon.ImageEventHandler):
    def __init__(
        self,
        recorder: Recorder,
        image_provider: ImageProvider,
        configuration: PySide6.QtQml.QQmlPropertyMap,
    ):
        super().__init__()
        self.recorder = recorder
        self.image_provider = image_provider

    def OnImageGrabbed(
        self,
        camera: pylon.InstantCamera,
        grabResult: pylon.GrabResult,
    ):
        if grabResult.GrabSucceeded():
            configuration.insert("queued_buffers", int(camera.NumQueuedBuffers.Value))
            timestamp = grabResult.GetTimeStamp()
            width = grabResult.GetWidth()
            height = grabResult.GetHeight()
            data = grabResult.GetImageBuffer()
            self.recorder.push(timestamp, width, height, data)
            self.image_provider.update_array(timestamp, width, height, data)


SUFFIX_AND_MULTIPLIER: list[tuple["str", float]] = [
    (" ms", 0.001),
    ("ms", 0.001),
    (" s", 1.0),
    ("s", 1.0),
]

FRAMERATE_KEYS: set[str] = {
    "width",
    "height",
    "framerate",
    "exposure",
}


def parse_duration(duration: str) -> float:
    for suffix, multiplier in SUFFIX_AND_MULTIPLIER:
        if duration.endswith(suffix):
            return float(duration[: -len(suffix)])
    raise Exception(f"parsing {duration=} failed (found no matching suffix)")


if __name__ == "__main__":
    configuration = PySide6.QtQml.QQmlPropertyMap()
    configuration.insert("recording_name", None)
    configuration.insert("recording_bytes", "0 B")
    configuration.insert("recording_frames", 0)
    configuration.insert("recording_duration", "00:00:00.000")
    configuration.insert("queued_buffers", 0)
    configuration.insert("maximum_queued_buffers", 0)
    configuration.insert("buffered_frames", "0 frames")
    configuration.insert("mode", 0)
    configuration.insert("circular_buffer_duration", "1 s")

    application = PySide6.QtGui.QGuiApplication(sys.argv)
    PySide6.QtGui.QFontDatabase.addApplicationFont(str(dirname / "RobotoMono.ttf"))
    monospace_font = PySide6.QtGui.QFontDatabase.font("Roboto Mono", "Regular", 12)
    monospace_font.setPixelSize(12)
    engine = PySide6.QtQml.QQmlApplicationEngine()
    engine.quit.connect(application.quit)
    image_provider = ImageProvider(configuration=configuration)
    engine.rootContext().setContextProperty("monospace_font", monospace_font)
    engine.rootContext().setContextProperty("configuration", configuration)
    engine.addImageProvider("camera", image_provider)
    engine.load("record.qml")
    recordings = dirname / "recordings"
    recordings.mkdir(exist_ok=True)
    application_globals = {
        "code": 0,
        "mode": 0,
        "circular_buffer_duration_s": 5.0,
    }
    with Recorder(configuration=configuration) as recorder:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.RegisterImageEventHandler(
            ImageEventHandler(
                recorder=recorder,
                image_provider=image_provider,
                configuration=configuration,
            ),
            pylon.RegistrationMode_Append,
            pylon.Cleanup_Delete,
        )
        camera.MaxNumQueuedBuffer.Value = 1024
        camera.Open()
        if camera.Width.Value > 640:
            camera.Width.Value = 640
            camera.OffsetX.Value = 16
        else:
            camera.OffsetX.Value = 16
            camera.Width.Value = 640
        if camera.Height.Value > 480:
            camera.Height.Value = 480
            camera.OffsetY.Value = 8
        else:
            camera.OffsetY.Value = 8
            camera.Height.Value = 480
        camera.ExposureAuto.Value = "Off"
        camera.GainAuto.Value = "Off"
        camera.PixelFormat.Value = "Mono10p"
        camera.ExposureTime.Value = 1000.0  # µs
        camera.Gain.Value = 0.0
        camera.AcquisitionFrameRateEnable.Value = True
        camera.AcquisitionFrameRate.Value = 500
        configuration.insert("calculated_framerate", camera.ResultingFrameRate.Value)
        configuration.insert(
            "maximum_queued_buffers", int(camera.MaxNumQueuedBuffer.Value)
        )

        def on_configuration_update(key: str, value: typing.Any):
            if key == "width":
                camera.StopGrabbing()
                if camera.Width.Value > int(value):
                    camera.Width.Value = int(value)
                    camera.OffsetX.Value = int(round((656 - int(value)) / 32) * 16)
                else:
                    camera.OffsetX.Value = int(round((656 - int(value)) / 32) * 16)
                    camera.Width.Value = int(value)
                camera.StartGrabbing(
                    pylon.GrabStrategy_OneByOne,
                    pylon.GrabLoop_ProvidedByInstantCamera,
                )
            elif key == "height":
                camera.StopGrabbing()
                if camera.Height.Value > int(value):
                    camera.Height.Value = int(value)
                    camera.OffsetY.Value = (496 - int(value)) // 2
                else:
                    camera.OffsetY.Value = (496 - int(value)) // 2
                    camera.Height.Value = int(value)
                camera.StartGrabbing(
                    pylon.GrabStrategy_OneByOne,
                    pylon.GrabLoop_ProvidedByInstantCamera,
                )
            elif key == "framerate":
                camera.AcquisitionFrameRate.Value = float(value) / 10.0
            elif key == "exposure":
                camera.ExposureTime.Value = float(value)
            elif key == "gain":
                camera.Gain.Value = float(value) / 100.0
            elif key == "calculated_framerate":
                pass
            elif key == "measured_framerate":
                pass
            elif key == "start_recording":
                exposure = camera.ExposureTime.Value
                gain = camera.Gain.Value
                recording_name = (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                    .replace(":", "-")
                    + ".basler"
                )
                recorder.start(
                    path=recordings / recording_name,
                    width=camera.Width.Value,
                    height=camera.Height.Value,
                    exposure=exposure,
                    gain=gain,
                )
                configuration.insert("recording_name", recording_name)
            elif key == "stop_recording":
                recorder.stop()
                configuration.insert("recording_name", None)
            elif key == "mode":
                application_globals["mode"] = value
                if value == 0:
                    recorder.direct_mode()
                elif value == 1:
                    recorder.circular_buffer(
                        length=round(
                            application_globals["circular_buffer_duration_s"]
                            * camera.ResultingFrameRate.Value
                        )
                    )
                else:
                    raise Exception(f'unexpected {value=} for key "mode"')
            elif key == "circular_buffer_duration":
                application_globals["circular_buffer_duration_s"] = parse_duration(
                    value
                )
                if application_globals["mode"] == 1:
                    recorder.circular_buffer(
                        length=round(
                            application_globals["circular_buffer_duration_s"]
                            * camera.ResultingFrameRate.Value
                        )
                    )
            else:
                print(f"unknown {key=} with value {value}")
            configuration.insert(
                "calculated_framerate", camera.ResultingFrameRate.Value
            )

            if key in FRAMERATE_KEYS:
                if application_globals["mode"] == 1:
                    recorder.circular_buffer(
                        length=round(
                            application_globals["circular_buffer_duration_s"]
                            * camera.ResultingFrameRate.Value
                        )
                    )

        configuration.valueChanged.connect(on_configuration_update)
        camera.StartGrabbing(
            pylon.GrabStrategy_OneByOne,
            pylon.GrabLoop_ProvidedByInstantCamera,
        )
        application_globals["code"] = application.exec()
        camera.Close()
    sys.exit(application_globals["code"])
