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

import transform

dirname = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass
class Output:
    path: pathlib.Path
    pixel_format: transform.PixelFormat
    width: int
    height: int
    exposure: float
    gain: float


def size_to_string(size: int) -> str:
    if size < 1000:
        return f"{size} B"
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
                                "<BHHdd",
                                transform.pixel_format_to_id(queue_output.pixel_format),
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
                                "<BHHdd",
                                transform.pixel_format_to_id(new_output.pixel_format),
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
        pixel_format: transform.PixelFormat,
        width: int,
        height: int,
        exposure: float,
        gain: float,
    ):
        self.output = Output(
            path=path,
            pixel_format=pixel_format,
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
    def __init__(
        self,
        pixel_format: transform.PixelFormat,
        default_width: int,
        default_height: int,
        configuration: PySide6.QtQml.QQmlPropertyMap,
    ):
        super().__init__(PySide6.QtQml.QQmlImageProviderBase.ImageType.Image)
        self.pixel_format = pixel_format
        self.default_width = default_width
        self.default_height = default_height
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

        self.unpacker = transform.Unpacker()

        if pixel_format == "BayerRG12p":
            self.bit_depth = 12
            self.demosaicer = transform.Demosaicer()
        elif pixel_format == "Mono10p":
            self.bit_depth = 10
            self.demosaicer = None
        else:
            return Exception(f"unknown pixel format {pixel_format}")

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
            size.setWidth(self.default_width)
            size.setHeight(self.default_height)
            return PySide6.QtGui.QImage(
                self.default_width,
                self.default_height,
                PySide6.QtGui.QImage.Format.Format_Grayscale16,
            )
        unpacked_array = self.unpacker.unpack(
            width=self.request_width,
            height=self.request_height,
            data=self.request_array,
            pixel_format=pixel_format,
        )
        size.setWidth(self.request_width)
        size.setHeight(self.request_height)

        if self.demosaicer is None:
            image = PySide6.QtGui.QImage(
                unpacked_array.tobytes(),
                self.request_width,
                self.request_height,
                PySide6.QtGui.QImage.Format.Format_Grayscale16,
            )
        else:
            image = PySide6.QtGui.QImage(
                self.demosaicer.demosaicize(
                    unpacked_array.reshape((self.request_height, self.request_width))
                ).tobytes(),
                self.request_width,
                self.request_height,
                PySide6.QtGui.QImage.Format.Format_RGB888,
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

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    available_pixel_formats = camera.PixelFormat.GetSymbolics()
    if "BayerRG12p" in available_pixel_formats:
        pixel_format: transform.PixelFormat = "BayerRG12p"
    elif "Mono10p":
        pixel_format: transform.PixelFormat = "Mono10p"
    else:
        raise Exception(
            f"the camera's pixel format is not supported by this recorder (the supported formats are BayerRG12p and Mono10p but the camera only supports {available_pixel_formats})"
        )

    application = PySide6.QtGui.QGuiApplication(sys.argv)
    PySide6.QtGui.QFontDatabase.addApplicationFont(str(dirname / "RobotoMono.ttf"))
    monospace_font = PySide6.QtGui.QFontDatabase.font("Roboto Mono", "Regular", 12)
    monospace_font.setPixelSize(12)
    engine = PySide6.QtQml.QQmlApplicationEngine()
    engine.quit.connect(application.quit)
    image_provider = ImageProvider(
        pixel_format=pixel_format,
        default_width=camera.Width.Max,
        default_height=camera.Height.Max,
        configuration=configuration,
    )
    engine.rootContext().setContextProperty("monospace_font", monospace_font)
    engine.rootContext().setContextProperty("configuration", configuration)
    engine.addImageProvider("camera", image_provider)
    engine.setInitialProperties(
        {
            "maximum_width": camera.Width.Max,
            "width_increment": camera.Width.Inc,
            "maximum_height": camera.Height.Max,
            "height_increment": camera.Height.Inc,
            "maximum_framerate": camera.AcquisitionFrameRate.Max * 10,
            "minimum_exposure": camera.ExposureTime.Min,
            "maximum_exposure": camera.ExposureTime.Max,
            "minimum_gain": camera.Gain.Min * 100,
            "maximum_gain": camera.Gain.Max * 100,
        }
    )
    configuration.insert("width", camera.Width.Max)
    configuration.insert("height", camera.Height.Max)
    configuration.insert("x_offset", 0)
    configuration.insert("y_offset", 0)
    engine.load("record.qml")
    recordings = dirname / "recordings"
    recordings.mkdir(exist_ok=True)
    application_globals = {
        "code": 0,
        "mode": 0,
        "circular_buffer_duration_s": 5.0,
    }
    with Recorder(configuration=configuration) as recorder:
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
        camera.OffsetX.Value = 0
        camera.Width.Value = camera.Width.Max
        camera.OffsetY.Value = 0
        camera.Height.Value = camera.Height.Max
        camera.ExposureAuto.Value = "Off"
        camera.GainAuto.Value = "Off"
        if pixel_format == "BayerRG12p":
            camera.BalanceWhiteAuto.Value = "Off"
        camera.PixelFormat.Value = pixel_format
        camera.ExposureTime.Value = 3000.0  # µs
        camera.Gain.Value = 0.0
        camera.AcquisitionFrameRateEnable.Value = True
        camera.AcquisitionFrameRate.Value = 1000
        configuration.insert("calculated_framerate", camera.ResultingFrameRate.Value)
        configuration.insert(
            "maximum_queued_buffers", int(camera.MaxNumQueuedBuffer.Value)
        )

        def on_configuration_update(key: str, value: typing.Any):
            if key == "width":
                if int(value) % camera.Width.Inc == 0:
                    camera.StopGrabbing()
                    if camera.Width.Value > int(value):
                        camera.Width.Value = int(value)
                        camera.OffsetX.Value = max(
                            int(
                                round(
                                    (camera.Width.Max - int(value))
                                    / (camera.Width.Inc * 2)
                                )
                                * camera.Width.Inc
                            ),
                            0,
                        )
                    else:
                        camera.OffsetX.Value = max(
                            int(
                                round(
                                    (camera.Width.Max - int(value))
                                    / (camera.Width.Inc * 2)
                                )
                                * camera.Width.Inc
                            ),
                            0,
                        )
                        camera.Width.Value = int(value)
                    configuration.insert("x_offset", camera.OffsetX.Value)
                    camera.StartGrabbing(
                        pylon.GrabStrategy_OneByOne,
                        pylon.GrabLoop_ProvidedByInstantCamera,
                    )
                else:
                    print(f"Warning: Width must be a multiple of {camera.Width.Inc}")
                    configuration.insert("width", camera.Width.Value)
                    configuration.insert("x_offset", camera.OffsetX.Value)
            elif key == "height":
                if int(value) % camera.Height.Inc == 0:
                    camera.StopGrabbing()
                    if camera.Height.Value > int(value):
                        camera.Height.Value = int(value)
                        camera.OffsetY.Value = max(
                            int(
                                round(
                                    (camera.Height.Max - int(value))
                                    / (camera.Height.Inc * 2)
                                )
                                * camera.Height.Inc
                            ),
                            0,
                        )
                    else:
                        camera.OffsetY.Value = max(
                            int(
                                round(
                                    (camera.Height.Max - int(value))
                                    / (camera.Height.Inc * 2)
                                )
                                * camera.Height.Inc
                            ),
                            0,
                        )
                        camera.Height.Value = int(value)
                    configuration.insert("y_offset", camera.OffsetY.Value)
                    camera.StartGrabbing(
                        pylon.GrabStrategy_OneByOne,
                        pylon.GrabLoop_ProvidedByInstantCamera,
                    )
                else:
                    print(f"Warning: Height must be a multiple of {camera.Height.Inc}")
                    configuration.insert("height", camera.Height.Value)
                    configuration.insert("y_offset", camera.OffsetY.Value)
            elif key == "x_offset":
                if int(value) % camera.Width.Inc == 0:
                    new_width = camera.Width.Value + int(value)
                    if new_width <= camera.Width.Max:
                        camera.StopGrabbing()
                        camera.OffsetX.Value = int(value)
                        camera.StartGrabbing(
                            pylon.GrabStrategy_OneByOne,
                            pylon.GrabLoop_ProvidedByInstantCamera,
                        )
                    else:
                        configuration.insert("x_offset", camera.OffsetX.Value)
                else:
                    print(f"Warning: X offset must be a multiple of {camera.Width.Inc}")
                    configuration.insert("width", camera.Width.Value)
                    configuration.insert("x_offset", camera.OffsetX.Value)
            elif key == "y_offset":
                if int(value) % camera.Height.Inc == 0:
                    new_height = camera.Height.Value + int(value)
                    if new_height <= camera.Height.Max:
                        camera.StopGrabbing()
                        camera.OffsetY.Value = int(value)
                        camera.StartGrabbing(
                            pylon.GrabStrategy_OneByOne,
                            pylon.GrabLoop_ProvidedByInstantCamera,
                        )
                    else:
                        configuration.insert("y_offset", camera.OffsetX.Value)
                else:
                    print(
                        f"Warning: Y offset must be a multiple of {camera.Height.Inc}"
                    )
                    configuration.insert("height", camera.Height.Value)
                    configuration.insert("y_offset", camera.OffsetY.Value)
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
                    pixel_format=pixel_format,
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
