use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

#[pyfunction]
fn unpack_10bit(
    input_array: &pyo3::Bound<'_, pyo3::types::PyByteArray>,
    output_frame: &pyo3::Bound<'_, numpy::PyArray2<u16>>,
) -> PyResult<()> {
    if !output_frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the output frame's memory must be contiguous"
        )));
    }
    let mut readwrite_output_frame = output_frame.readwrite();
    let output_dimensions = readwrite_output_frame.as_array_mut().dim();
    if input_array.len() * 4 != output_dimensions.0 * output_dimensions.1 * 5 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the input array length (in bytes) must be 1.25 times the length of the output frame (in pixels)"
        )));
    }
    let input_slice = unsafe { input_array.as_bytes() };
    let output_slice = readwrite_output_frame
        .as_slice_mut()
        .expect("the frame is contiguous");

    let mut output_index = 0;


    // self._unpack_bytes(0, 0b11111111, 0, 1, 0b11, 8, 0)
    // self._unpack_bytes(1, 0b11111111, -2, 2, 0b1111, 6, 1)
    // self._unpack_bytes(2, 0b11111111, -4, 3, 0b111111, 4, 2)
    // self._unpack_bytes(3, 0b11111111, -6, 4, 0b11111111, 2, 3)

    // @DEV todo

    for input_index in (0..input_array.len()).step_by(5) {
        output_slice[output_index] = u16::from_le_bytes([
            input_slice[input_index],
            input_slice[input_index + 1] & 0b11,
        ]) << 6;
        output_index += 1;

        output_slice[output_index] = u16::from_le_bytes([
            (input_slice[input_index + 1] >> 2) | ((input_slice[input_index + 2] & 0b11) << 2),
            input_slice[input_index + 2] >> 4,
        ]) << 4;
        output_index += 1;
    }
    Ok(())
}



#[pyfunction]
fn unpack_12bit(
    input_array: &pyo3::Bound<'_, pyo3::types::PyByteArray>,
    output_frame: &pyo3::Bound<'_, numpy::PyArray2<u16>>,
) -> PyResult<()> {
    if !output_frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the output frame's memory must be contiguous"
        )));
    }
    let mut readwrite_output_frame = output_frame.readwrite();
    let output_dimensions = readwrite_output_frame.as_array_mut().dim();
    if input_array.len() * 2 != output_dimensions.0 * output_dimensions.1 * 3 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the input array length (in bytes) must be 1.5 times the length of the output frame (in pixels)"
        )));
    }
    let input_slice = unsafe { input_array.as_bytes() };
    let output_slice = readwrite_output_frame
        .as_slice_mut()
        .expect("the frame is contiguous");

    let mut output_index = 0;

    for input_index in (0..input_array.len()).step_by(3) {
        output_slice[output_index] = u16::from_le_bytes([
            input_slice[input_index],
            input_slice[input_index + 1] & 0b1111,
        ]) << 4;
        output_index += 1;
        output_slice[output_index] = u16::from_le_bytes([
            (input_slice[input_index + 1] >> 4) | ((input_slice[input_index + 2] & 0b1111) << 4),
            input_slice[input_index + 2] >> 4,
        ]) << 4;
        output_index += 1;
    }
    Ok(())
}

fn from_one(input_slice: &[u16], width: usize, xy: (usize, usize)) -> u8 {
    (input_slice[xy.0 + xy.1 * width] >> 8) as u8
}

fn from_two(input_slice: &[u16], width: usize, xy0: (usize, usize), xy1: (usize, usize)) -> u8 {
    (((input_slice[xy0.0 + xy0.1 * width] >> 1) + (input_slice[xy1.0 + xy1.1 * width] >> 1)) >> 8)
        as u8
}

fn from_three(
    input_slice: &[u16],
    width: usize,
    xy0: (usize, usize),
    xy1: (usize, usize),
    xy2: (usize, usize),
) -> u8 {
    (((input_slice[xy0.0 + xy0.1 * width] >> 1)
        + (input_slice[xy1.0 + xy1.1 * width] >> 2)
        + (input_slice[xy2.0 + xy2.1 * width] >> 2))
        >> 8) as u8
}

fn from_four(
    input_slice: &[u16],
    width: usize,
    xy0: (usize, usize),
    xy1: (usize, usize),
    xy2: (usize, usize),
    xy3: (usize, usize),
) -> u8 {
    (((input_slice[xy0.0 + xy0.1 * width] >> 2)
        + (input_slice[xy1.0 + xy1.1 * width] >> 2)
        + (input_slice[xy2.0 + xy2.1 * width] >> 2)
        + (input_slice[xy3.0 + xy3.1 * width] >> 2))
        >> 8) as u8
}

#[pyfunction]
fn demosaicize(
    input_frame: &pyo3::Bound<'_, numpy::PyArray2<u16>>,
    output_frame: &pyo3::Bound<'_, numpy::PyArray3<u8>>,
) -> PyResult<()> {
    if !input_frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the input frame's memory must be contiguous"
        )));
    }
    if !output_frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the output frame's memory must be contiguous"
        )));
    }
    let readonly_input_frame = input_frame.readonly();
    let input_dimensions = readonly_input_frame.as_array().dim();
    let mut readwrite_output_frame = output_frame.readwrite();
    let output_dimensions = readwrite_output_frame.as_array_mut().dim();
    if input_dimensions.0 != output_dimensions.0 || input_dimensions.1 != output_dimensions.1 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the input and output frames must have the same dimensions"
        )));
    }
    if input_dimensions.0 == 0
        || input_dimensions.1 == 0
        || input_dimensions.0 % 2 != 0
        || input_dimensions.1 % 2 != 0
    {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the width and height must be non-zero and even"
        )));
    }
    let input_slice = readonly_input_frame
        .as_slice()
        .expect("the frame is contiguous");
    let output_slice = readwrite_output_frame
        .as_slice_mut()
        .expect("the frame is contiguous");
    let width = input_dimensions.1;
    let height = input_dimensions.0;

    // top-left
    {
        let index = 0;
        output_slice[index + 0] = from_one(input_slice, width, (0, 0));
        output_slice[index + 1] = from_two(input_slice, width, (1, 0), (0, 1));
        output_slice[index + 2] = from_one(input_slice, width, (1, 1));
    }

    // top
    for x in (1..width - 1).step_by(2) {
        let index = x * 3;
        output_slice[index + 0] = from_two(input_slice, width, (x - 1, 0), (x + 1, 0));
        output_slice[index + 1] = from_one(input_slice, width, (x, 0));
        output_slice[index + 2] = from_one(input_slice, width, (x, 1));
    }
    for x in (2..width - 1).step_by(2) {
        let index = x * 3;
        output_slice[index + 0] = from_one(input_slice, width, (x, 0));
        output_slice[index + 1] = from_three(input_slice, width, (x, 1), (x - 1, 0), (x + 1, 0));
        output_slice[index + 2] = from_two(input_slice, width, (x - 1, 1), (x + 1, 1));
    }

    // top-right
    {
        let index = (width - 1) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (width - 2, 0));
        output_slice[index + 1] = from_one(input_slice, width, (width - 1, 0));
        output_slice[index + 2] = from_one(input_slice, width, (width - 1, 1));
    }

    // left
    for y in (1..height - 1).step_by(2) {
        let index = (y * width) * 3;
        output_slice[index + 0] = from_two(input_slice, width, (0, y - 1), (0, y + 1));
        output_slice[index + 1] = from_one(input_slice, width, (0, y));
        output_slice[index + 2] = from_one(input_slice, width, (1, y));
    }
    for y in (2..height - 1).step_by(2) {
        let index = (y * width) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (0, y));
        output_slice[index + 1] = from_three(input_slice, width, (1, y), (0, y - 1), (0, y + 1));
        output_slice[index + 2] = from_two(input_slice, width, (1, y - 1), (1, y + 1));
    }

    // inner
    for y in (1..height - 1).step_by(2) {
        for x in (1..width - 1).step_by(2) {
            let index = (x + y * width) * 3;
            output_slice[index + 0] = from_four(
                input_slice,
                width,
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y + 1),
            );
            output_slice[index + 1] = from_four(
                input_slice,
                width,
                (x, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x, y + 1),
            );
            output_slice[index + 2] = from_one(input_slice, width, (x, y));
        }
        for x in (2..width - 1).step_by(2) {
            let index = (x + y * width) * 3;
            output_slice[index + 0] = from_two(input_slice, width, (x, y - 1), (x, y + 1));
            output_slice[index + 1] = from_one(input_slice, width, (x, y));
            output_slice[index + 2] = from_two(input_slice, width, (x - 1, y), (x + 1, y));
        }
    }
    for y in (2..height - 1).step_by(2) {
        for x in (1..width - 1).step_by(2) {
            let index = (x + y * width) * 3;
            output_slice[index + 0] = from_two(input_slice, width, (x - 1, y), (x + 1, y));
            output_slice[index + 1] = from_one(input_slice, width, (x, y));
            output_slice[index + 2] = from_two(input_slice, width, (x, y - 1), (x, y + 1));
        }
        for x in (2..width - 1).step_by(2) {
            let index = (x + y * width) * 3;
            output_slice[index + 0] = from_one(input_slice, width, (x, y));
            output_slice[index + 1] = from_four(
                input_slice,
                width,
                (x, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x, y + 1),
            );
            output_slice[index + 2] = from_four(
                input_slice,
                width,
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y + 1),
            );
        }
    }

    // right
    for y in (1..height - 1).step_by(2) {
        let index = (width - 1 + y * width) * 3;
        output_slice[index + 0] =
            from_two(input_slice, width, (width - 2, y - 1), (width - 2, y + 1));
        output_slice[index + 1] = from_three(
            input_slice,
            width,
            (width - 2, y),
            (width - 1, y - 1),
            (width - 1, y + 1),
        );
        output_slice[index + 2] = from_one(input_slice, width, (width - 1, y));
    }
    for y in (2..height - 1).step_by(2) {
        let index = (width - 1 + y * width) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (width - 2, y));
        output_slice[index + 1] = from_one(input_slice, width, (width - 1, y));
        output_slice[index + 2] =
            from_two(input_slice, width, (width - 1, y - 1), (width - 1, y + 1));
    }

    // bottom-left
    {
        let index = ((height - 1) * width) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (0, height - 2));
        output_slice[index + 1] = from_one(input_slice, width, (0, height - 1));
        output_slice[index + 2] = from_one(input_slice, width, (1, height - 1));
    }

    // bottom
    for x in (1..width - 1).step_by(2) {
        let index = (x + (height - 1) * width) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (x, height - 2));
        output_slice[index + 1] = from_one(input_slice, width, (x, height - 1));
        output_slice[index + 2] =
            from_two(input_slice, width, (x - 1, height - 1), (x + 1, height - 1));
    }
    for x in (2..width - 1).step_by(2) {
        let index = (x + (height - 1) * width) * 3;
        output_slice[index + 0] =
            from_two(input_slice, width, (x - 1, height - 2), (x + 1, height - 2));
        output_slice[index + 1] = from_three(
            input_slice,
            width,
            (x, height - 2),
            (x - 1, height - 1),
            (x + 1, height - 1),
        );
        output_slice[index + 2] = from_one(input_slice, width, (x, height - 1));
    }

    // bottom-right
    {
        let index = (width - 1 + (height - 1) * width) * 3;
        output_slice[index + 0] = from_one(input_slice, width, (width - 2, height - 2));
        output_slice[index + 1] = from_two(
            input_slice,
            width,
            (width - 1, height - 2),
            (width - 2, height - 1),
        );
        output_slice[index + 2] = from_one(input_slice, width, (width - 1, height - 1));
    }
    Ok(())
}

#[pymodule]
#[pyo3(name = "extension")]
fn transform(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(unpack_12bit, module)?)?;
    module.add_function(wrap_pyfunction!(demosaicize, module)?)?;
    Ok(())
}
