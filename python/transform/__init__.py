import typing

import time  # @DEV

import numpy
import scipy.ndimage

from . import extension

PixelFormat = typing.Literal["BayerRG12p", "Mono10p"]


def pixel_format_to_id(pixel_format: PixelFormat) -> int:
    if pixel_format == "BayerRG12p":
        return 0
    if pixel_format == "Mono10p":
        return 1
    return Exception(f"unknown pixel format {pixel_format}")


def id_to_pixel_format(id: int) -> PixelFormat:
    if id == 0:
        return "BayerRG12p"
    if id == 1:
        return "Mono10p"
    raise Exception(f"unknown pixel format id {id}")


class Unpacker10bit:
    def __init__(self):
        self.packed_array = numpy.zeros(0, dtype=numpy.uint8)
        self.unpack_buffer = numpy.zeros(0, dtype=numpy.uint16)
        self.unpacked_array = numpy.zeros(0, dtype=numpy.uint16)
        self.width = 0
        self.height = 0

    def unpack(
        self,
        width: int,
        height: int,
        data: bytearray,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        self.width = width
        self.height = height
        if len(self.packed_array) != len(data):
            self.packed_array = numpy.zeros(len(data), dtype=numpy.uint8)
        numpy.copyto(self.packed_array, numpy.frombuffer(data, dtype=numpy.uint8))
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
            self.unpacked_array[output_offset::4] = self.packed_array[byte_0_offset::5]
        else:
            numpy.bitwise_and(
                self.packed_array[byte_0_offset::5],
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
            self.unpack_buffer[::] = self.packed_array[byte_1_offset::5]
        else:
            numpy.bitwise_and(
                self.packed_array[byte_1_offset::5],
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


class Unpacker12bit:
    def __init__(self):
        self.output_frame = numpy.zeros((0, 0), dtype=numpy.uint16)

    def unpack(
        self,
        width: int,
        height: int,
        data: bytearray,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        if self.output_frame.shape[0] != height or self.output_frame.shape[1] != width:
            self.output_frame = numpy.zeros((height, width), dtype=numpy.uint16)
        extension.unpack_12bit(input_array=data, output_frame=self.output_frame)
        return self.output_frame


class Unpacker:
    def __init__(self):
        self.inner_12_bits: typing.Optional[Unpacker12bit] = None
        self.inner_10_bits: typing.Optional[Unpacker10bit] = None

    def unpack(
        self,
        width: int,
        height: int,
        data: bytearray,
        pixel_format: PixelFormat,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        if pixel_format == "BayerRG12p":
            if self.inner_12_bits is None:
                self.inner_12_bits = Unpacker12bit()
            return self.inner_12_bits.unpack(width=width, height=height, data=data)
        if pixel_format == "Mono10p":
            if self.inner_10_bits is None:
                self.inner_10_bits = Unpacker10bit()
            return self.inner_10_bits.unpack(width=width, height=height, data=data)
        else:
            raise Exception(f"unsupported pixel format {pixel_format}")


class Demosaicer:
    def __init__(self):
        self.output_frame = numpy.zeros((0, 0), dtype=numpy.uint8)

    def demosaicize(
        self,
        input_frame: numpy.ndarray,
    ) -> numpy.ndarray:
        if (
            input_frame.shape[0] != self.output_frame.shape[0]
            or input_frame.shape[1] != self.output_frame.shape[1]
        ):
            self.output_frame = numpy.zeros(
                (input_frame.shape[0], input_frame.shape[1], 3), dtype=numpy.uint8
            )
        extension.demosaicize(
            input_frame=input_frame,
            output_frame=self.output_frame,
        )
        return self.output_frame
