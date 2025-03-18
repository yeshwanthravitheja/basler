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
        self.width = 0
        self.height = 0

    def unpack(
        self,
        width: int,
        height: int,
        data: bytes,
    ) -> numpy.typing.NDArray[numpy.uint8]:
        self.width = width
        self.height = height
        self.array = bytearray(data)
        return numpy.frombuffer(self.array, dtype=numpy.uint8).reshape((height, width, 3))


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
        data_length = width * height * 3  # RGB888 format
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
                PIL.Image.fromarray(unpacked_array, mode="RGB").save(
                    str(write_output_path), format="png", compress_level=1
                )
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
                    )
                    frame[:, :, :] = unpacked_array
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
