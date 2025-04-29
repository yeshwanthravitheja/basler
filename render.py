import argparse
import json
import math
import os
import pathlib
import struct
import sys
import typing

import faery
import faery.frame_filter
import numpy
import numpy.typing
import PIL.Image

import pidng.core
import pidng.defs

import transform


def draw_progress_bar(columns: int, image_index: int, image_count: int):
    count_width = math.ceil(math.log10(image_count))
    prefix = f"{image_index: {count_width}} / {image_count}"
    progress_bar_width = columns - len(prefix) - 2
    if progress_bar_width > 2:
        sys.stdout.write(
            f"\r{prefix} {faery.display.generate_progress_bar(width=progress_bar_width, progress=image_index / image_count)}"
        )
    else:
        sys.stdout.write(f"\r{' ' * columns}\r{prefix}")
    if image_index == image_count:
        sys.stdout.write(f"\r{' ' * columns}\r")


def unpack_file(
    path: pathlib.Path,
    unpacker: typing.Optional[transform.Unpacker] = None,
    demosaicer: typing.Optional[transform.Demosaicer] = None,
    generate_png_frames: bool = True,
    generate_dng_frames: bool = True,
    generate_video: bool = True,
    force: bool = False,
):
    if unpacker is None:
        unpacker = transform.Unpacker()
    output_directory = path.parent / f"{path.stem}"
    output_directory.mkdir(exist_ok=True)
    if generate_png_frames:
        png_frames_output = output_directory / f"{path.stem}_png_frames"
        png_frames_output.mkdir(exist_ok=True)
    else:
        png_frames_output = None
    if generate_dng_frames:
        dng_frames_output = output_directory / f"{path.stem}_dng_frames"
        dng_frames_output.mkdir(exist_ok=True)
    else:
        dng_frames_output = None
    timestamps_list = []

    # count frames
    image_count = 0
    file_size = path.stat().st_size
    with open(path, "rb") as file:
        assert file.read(6) == b"BASLER"
        header = file.read(21)
        if len(header) < 21:
            return
        pixel_format_id, width, height, exposure, gain = struct.unpack("<BHHdd", header)
        pixel_format = transform.id_to_pixel_format(pixel_format_id)
        if pixel_format == "BayerRG12p":
            bit_depth = 12
        elif pixel_format == "Mono10p":
            bit_depth = 10
        else:
            return Exception(f"unknown pixel format {pixel_format}")
        data_length = width * height * bit_depth // 8
        while True:
            timestamp_data = file.read(8)
            if len(timestamp_data) < 8:
                break
            if file.tell() > file_size - data_length:
                break
            file.seek(data_length, 1)
            image_count += 1

    # render "raw" PNGs and raw DNGs
    if generate_png_frames or generate_dng_frames:
        print(f"{path.stem} – Render frames")
        columns = os.get_terminal_size().columns
        draw_progress_bar(
            columns=columns,
            image_index=0,
            image_count=image_count,
        )
        with open(path, "rb") as file:
            image_index = 0
            assert file.read(6) == b"BASLER"
            header = file.read(21)
            if len(header) < 21:
                return
            pixel_format_id, width, height, exposure, gain = struct.unpack(
                "<BHHdd", header
            )
            pixel_format = transform.id_to_pixel_format(pixel_format_id)
            if pixel_format == "BayerRG12p":
                bit_depth = 12
            elif pixel_format == "Mono10p":
                bit_depth = 10
            else:
                return Exception(f"unknown pixel format {pixel_format}")
            data_length = width * height * bit_depth // 8
            data = bytearray(data_length)
            while True:
                timestamp_data = file.read(8)
                if len(timestamp_data) < 8:
                    break
                timestamps_list.append(struct.unpack("<Q", timestamp_data)[0])
                if file.readinto(data) != data_length:
                    break
                unpacked_array: typing.Optional[numpy.typing.NDArray[numpy.uint16]] = (
                    None
                )
                if png_frames_output is not None:
                    png_output_path = (
                        png_frames_output / f"{path.stem}_{image_index:06d}.png"
                    )
                    if force or not png_output_path.is_file():
                        png_write_output_path = (
                            png_frames_output
                            / f"{path.stem}_{image_index:06d}.png.write"
                        )
                        unpacked_array = unpacker.unpack(
                            width=width,
                            height=height,
                            data=data,
                            pixel_format=pixel_format,
                        )
                        PIL.Image.frombytes(
                            mode="I;16",
                            size=(width, height),
                            data=unpacked_array.tobytes(),
                        ).save(
                            str(png_write_output_path), format="png", compress_level=1
                        )
                        png_write_output_path.replace(png_output_path)

                if dng_frames_output is not None:
                    dng_output_path = (
                        dng_frames_output / f"{path.stem}_{image_index:06d}.dng"
                    )
                    if force or not dng_output_path.is_file():
                        if unpacked_array is None:
                            unpacked_array = unpacker.unpack(
                                width=width,
                                height=height,
                                data=data,
                                pixel_format=pixel_format,
                            )
                        unpacked_array >>= 16 - bit_depth
                        tags = pidng.core.DNGTags()
                        tags.set(pidng.core.Tag.ImageWidth, width)  # type: ignore
                        tags.set(pidng.core.Tag.ImageLength, height)  # type: ignore
                        tags.set(pidng.core.Tag.TileWidth, width)  # type: ignore
                        tags.set(pidng.core.Tag.TileLength, height)  # type: ignore
                        tags.set(
                            pidng.core.Tag.Orientation,  # type: ignore
                            pidng.defs.Orientation.Horizontal,
                        )
                        tags.set(
                            pidng.core.Tag.PhotometricInterpretation,  # type: ignore
                            pidng.defs.PhotometricInterpretation.Color_Filter_Array,
                        )
                        tags.set(pidng.core.Tag.SamplesPerPixel, 1)  # type: ignore
                        tags.set(pidng.core.Tag.BitsPerSample, bit_depth)  # type: ignore
                        tags.set(pidng.core.Tag.CFARepeatPatternDim, [2, 2])  # type: ignore
                        tags.set(
                            pidng.core.Tag.CFAPattern,  # type: ignore
                            pidng.defs.CFAPattern.RGGB,
                        )
                        tags.set(pidng.core.Tag.WhiteLevel, ((1 << bit_depth) - 1))  # type: ignore
                        tags.set(
                            pidng.core.Tag.ColorMatrix1,  # type: ignore
                            [
                                [32404, 10000],
                                [-15371, 10000],
                                [-4985, 10000],
                                [-9692, 10000],
                                [18760, 10000],
                                [415, 10000],
                                [556, 10000],
                                [-2040, 10000],
                                [10572, 10000],
                            ],
                        )
                        tags.set(
                            pidng.core.Tag.CalibrationIlluminant1,  # type: ignore
                            pidng.defs.CalibrationIlluminant.D65,
                        )
                        tags.set(pidng.core.Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])  # type: ignore
                        tags.set(pidng.core.Tag.BaselineExposure, [[-150, 100]])  # type: ignore
                        tags.set(pidng.core.Tag.Make, "Basler")  # type: ignore
                        tags.set(pidng.core.Tag.Model, "-")  # type: ignore
                        tags.set(
                            pidng.core.Tag.DNGVersion,  # type: ignore
                            pidng.defs.DNGVersion.V1_4,
                        )
                        tags.set(
                            pidng.core.Tag.DNGBackwardVersion,  # type: ignore
                            pidng.defs.DNGVersion.V1_2,
                        )
                        tags.set(
                            pidng.core.Tag.PreviewColorSpace,  # type: ignore
                            pidng.defs.PreviewColorSpace.sRGB,
                        )
                        converter = pidng.core.RAW2DNG()
                        converter.options(tags, path="", compress=False)
                        converter.convert(unpacked_array, filename=str(dng_output_path))

                image_index += 1
                draw_progress_bar(
                    columns=columns,
                    image_index=image_index,
                    image_count=image_count,
                )

    # create configuration and timestamps_ns files
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

    # render mp4 video
    if generate_video:
        print(f"{path.stem} – Render video")
        columns = os.get_terminal_size().columns
        output_path = output_directory / f"{path.stem}_render.mp4"
        if force or not output_path.is_file():
            draw_progress_bar(
                columns=columns,
                image_index=0,
                image_count=image_count,
            )
            write_output_path = output_directory / f"{path.stem}_render.mp4.write"
            image_index = 0
            with open(path, "rb") as file:
                assert file.read(6) == b"BASLER"
                header = file.read(21)
                if len(header) < 21:
                    return
                pixel_format_id, width, height, exposure, gain = struct.unpack(
                    "<BHHdd", header
                )
                gray_frame: typing.Optional[numpy.ndarray] = None
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
                        if file.readinto(data) != data_length:
                            break
                        unpacked_array = unpacker.unpack(
                            width=width,
                            height=height,
                            data=data,
                            pixel_format=pixel_format,
                        ).reshape((height, width))
                        if pixel_format == "BayerRG12p":
                            if demosaicer is None:
                                demosaicer = transform.Demosaicer()
                            frame = demosaicer.demosaicize(unpacked_array)
                        elif pixel_format == "Mono10p":
                            if gray_frame is None:
                                gray_frame = numpy.zeros(
                                    (height, width, 3), dtype=numpy.uint8
                                )
                            numpy.right_shift(
                                unpacked_array,
                                8,
                                out=unpacked_array,
                                dtype=numpy.uint16,
                            )
                            gray_frame[:, :, 0] = unpacked_array
                            gray_frame[:, :, 1] = gray_frame[:, :, 0]
                            gray_frame[:, :, 2] = gray_frame[:, :, 0]
                            frame = gray_frame
                        else:
                            return Exception(f"unknown pixel format {pixel_format}")
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
                        draw_progress_bar(
                            columns=columns,
                            image_index=image_index,
                            image_count=image_count,
                        )
            write_output_path.rename(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-png-frames", action="store_true")
    parser.add_argument("--no-dng-frames", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    unpacker = transform.Unpacker()
    demosaicer = transform.Demosaicer()
    for path in sorted((faery.dirname / "recordings").iterdir()):
        if path.is_file() and path.suffix == ".basler":
            unpack_file(
                path=path,
                unpacker=unpacker,
                demosaicer=demosaicer,
                generate_png_frames=not args.no_png_frames,
                generate_dng_frames=not args.no_dng_frames,
                generate_video=not args.no_video,
                force=args.force,
            )
