import numpy.typing

def unpack_12bit(
    input_array: bytearray,
    output_frame: numpy.typing.NDArray[numpy.uint16],
) -> None: ...
def demosaicize(
    input_frame: numpy.typing.NDArray[numpy.uint16],
    output_frame: numpy.typing.NDArray[numpy.uint8],
) -> None: ...
