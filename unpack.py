import json
import math
import pathlib
import struct
import typing

import faery
import faery.frame_filter
import numpy
import numpy.typing
import PIL.Image


class Unpacker:
    def __init__(self):
        self.array: bytes = bytes([])
        self.unpack_buffer = numpy.zeros(0, dtype=numpy.uint16)
        self.unpacked_array = numpy.zeros(0, dtype=numpy.uint16)
        self.width = 0
        self.height = 0

    def unpack(
        self,
        width: int,
        height: int,
        data: bytes,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        self.width = width
        self.height = height
        self.array = bytearray(data)
        self._unpack()
        return self.unpacked_array

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
            self.unpacked_array[output_offset::4] = self.array[byte_0_offset::5]
        else:
            numpy.bitwise_and(
                self.array[byte_0_offset::5],
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
            self.unpack_buffer[::] = self.array[byte_1_offset::5]
        else:
            numpy.bitwise_and(
                self.array[byte_1_offset::5],
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
        if len(self.unpacked_array) != self.width * self.height:
            self.unpacked_array = numpy.zeros(
                self.width * self.height, dtype=numpy.uint16
            )
            self.unpack_buffer = numpy.zeros(
                self.width * self.height // 4, dtype=numpy.uint16
            )
        self._unpack_bytes(0, 0b11111111, 0, 1, 0b11, 8, 0)
        self._unpack_bytes(1, 0b11111111, -2, 2, 0b1111, 6, 1)
        self._unpack_bytes(2, 0b11111111, -4, 3, 0b111111, 4, 2)
        self._unpack_bytes(3, 0b11111111, -6, 4, 0b11111111, 2, 3)
        numpy.bitwise_left_shift(self.unpacked_array, 6, out=self.unpacked_array)


def unpack_file(
    path: pathlib.Path,
    unpacker: typing.Optional[Unpacker] = None,
    force: bool = False,
):
    if unpacker is None:
        unpacker = Unpacker()
    output_directory = path.parent / f"{path.stem}"
    output_directory.mkdir(exist_ok=True)
    frames_output = output_directory / f"{path.stem}_frames"
    frames_output.mkdir(exist_ok=True)
    timestamps_list = []
    with open(path, "rb") as file:
        image_index = 0
        assert file.read(6) == b"BASLER"
        header = file.read(20)
        if len(header) < 20:
            return
        width, height, exposure, gain = struct.unpack("<HHdd", header)
        data_length = width * height * 10 // 8
        while True:
            timestamp_data = file.read(8)
            if len(timestamp_data) < 8:
                break
            timestamps_list.append(struct.unpack("<Q", timestamp_data)[0])
            data = file.read(data_length)
            if len(data) != data_length:
                break
            output_path = frames_output / f"{path.stem}_{image_index:06d}.png"
            image_index += 1
            if force or not output_path.is_file():
                write_output_path = (
                    frames_output / f"{path.stem}_{image_index:06d}.png.write"
                )
                unpacked_array = unpacker.unpack(width=width, height=height, data=data)
                PIL.Image.frombytes(
                    mode="I;16",
                    size=(width, height),
                    data=unpacked_array.tobytes(),
                ).save(str(write_output_path), format="png", compress_level=1)
                write_output_path.replace(output_path)
    mean_timestamp_delta = numpy.mean(
        numpy.diff(numpy.array(timestamps_list, dtype=numpy.uint64))
    )
    framerate = float(1e9 / mean_timestamp_delta)
    duration = 0.0
    if len(timestamps_list) > 1:
        duration = timestamps_list[-1] / 1e3 - timestamps_list[0] / 1e3
    duration += exposure
    with open(
        output_directory / f"{path.stem}_configuration.json", "w"
    ) as configuration_file:
        json.dump(
            {
                "width": width,
                "height": height,
                "exposure_us": exposure,
                "gain": gain,
                "frame_rate_hz": framerate,
                "frame_count": len(timestamps_list),
                "duration": (int(round(duration)) * faery.us).to_timecode(),
            },
            configuration_file,
            indent=4,
        )
    with open(
        output_directory / f"{path.stem}_timestamps_ns.json", "w"
    ) as timestamps_file:
        json.dump(timestamps_list, timestamps_file)
    output_path = output_directory / f"{path.stem}_render.mp4"
    if force or not output_path.is_file():
        write_output_path = output_directory / f"{path.stem}_render.mp4.write"
        image_index = 0
        with open(path, "rb") as file:
            assert file.read(6) == b"BASLER"
            header = file.read(20)
            if len(header) < 20:
                return
            width, height, exposure, gain = struct.unpack("<HHdd", header)
            frame = numpy.zeros((height, width, 3), dtype=numpy.uint8)
            factor = min(
                math.ceil(4800 / width),
                math.ceil(3600 / height),
                max(1.0, math.ceil(960 / width), math.ceil(720 / height)),
            )
            dimensions = (
                int(round(width * factor)),
                int(round(height * factor)),
            )
            SPEED_UP_PRECISION: float = 1e-3
            speed_up = 60.0 / framerate
            if (
                speed_up < 1.0 + SPEED_UP_PRECISION
                and speed_up > 1.0 - SPEED_UP_PRECISION
            ):
                speedup_label = "Real-time"
            elif speed_up < 1.0:
                inverse_speedup = 1.0 / speed_up
                speedup_label = f"× 1/{faery.frame_filter.number_to_string(inverse_speedup, SPEED_UP_PRECISION)}"
            else:
                speedup_label = f"× {faery.frame_filter.number_to_string(speed_up, SPEED_UP_PRECISION)}"
            color = faery.color.color_to_ints("#FFFFFF")
            with faery.mp4.Encoder(
                path=write_output_path,
                dimensions=dimensions,
                frame_rate=60.0,
                crf=17.0,
                preset="medium",
                tune="none",
                profile="baseline",
            ) as encoder:
                while True:
                    timestamp_data = file.read(8)
                    if len(timestamp_data) < 8:
                        break
                    timestamp = (
                        struct.unpack("<Q", timestamp_data)[0] - timestamps_list[0]
                    )
                    data = file.read(data_length)
                    if len(data) != data_length:
                        break
                    unpacked_array = unpacker.unpack(
                        width=width,
                        height=height,
                        data=data,
                    ).reshape((height, width))
                    frame[:, :, 0] = unpacked_array >> 8
                    frame[:, :, 1] = frame[:, :, 0]
                    frame[:, :, 2] = frame[:, :, 0]
                    output_frame = faery.image.resize(
                        frame,
                        dimensions,
                        sampling_filter="nearest",
                    )
                    faery.image.annotate(
                        frame=output_frame,
                        text=(int(round(timestamp / 1e3)) * faery.us).to_timecode(),
                        x=21,
                        y=15,
                        size=30,
                        color=color,
                    )
                    faery.image.annotate(
                        frame=output_frame,
                        text=speedup_label,
                        x=21,
                        y=15 + round(30 * 1.2),
                        size=30,
                        color=color,
                    )
                    encoder.write(output_frame)
                    image_index += 1
        write_output_path.rename(output_path)


if __name__ == "__main__":
    unpacker = Unpacker()
    for path in sorted((faery.dirname / "recordings").iterdir()):
        if path.is_file() and path.suffix == ".basler":
            unpack_file(path=path, unpacker=unpacker)
