from math import isnan, isinf
from endia import Array
from endia.functional import *


@always_inline
fn to_torch(arg: Array) raises -> PythonObject:
    """
    Converts an endia Array to a torch tensor.
    """
    if arg.is_complex():
        # print("\narg:")
        # print(arg)
        var real = real(arg)
        var imag = imag(arg)
        # print("\nreal:")
        # print(real)
        # print("\nimag:")
        # print(imag)
        var torch = Python.import_module("torch")
        var torch_real = to_torch_tensor(real).requires_grad_(
            arg.requires_grad()
        )
        var torch_imag = to_torch_tensor(imag).requires_grad_(
            arg.requires_grad()
        )
        var res = torch.complex(torch_real, torch_imag)
        return res
    return to_torch_tensor(arg)


@always_inline
fn to_torch_tensor(arg: Array) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var shape = arg.shape()
    var size = 1
    for i in range(shape.size):
        size *= shape[i]
    var torch_shape = PythonObject([])
    for i in range(arg.ndim()):
        torch_shape.append(shape[i])
    var res = torch.zeros(size=torch_shape).to(torch.float64)

    var flattened = res.flatten()
    for i in range(size):
        flattened[i] = arg.load(i)

    if arg.requires_grad():
        res.requires_grad = True

    return res


@always_inline
fn is_close_to_tensor(
    arr: Array, arr_torch: PythonObject, rtol: Float32
) raises -> Bool:
    """
    Asserts that the values in the endia Array and the torch tensor are equal.
    """
    var shape = arr.shape()
    var size = 1
    for i in range(shape.size):
        size *= shape[i]
    var flattened = arr_torch.flatten()
    var wrong_occurences: Int = 0
    for i in range(size):
        var arr_val = arr.load(i)
        var torch_val = flattened[i].to_float64().cast[dtype]()
        if not isnan(arr_val) and not isinf(arr_val):
            var rel_diff = arr_val / torch_val
            if rel_diff < 1 - rtol or rel_diff > 1 + rtol:
                wrong_occurences += 1
                # print(
                #     "Incorrect value at index",
                #     i,
                #     " - endia_val =",
                #     arr.load(i),
                #     " - torch_val =",
                #     flattened[i].to_float64().cast[dtype](),
                # )

    if wrong_occurences > 0:
        # print("Warning: Number of wrong occurences: ", wrong_occurences, "out of", size, "total elements at relative tolerance", rtol, "!")
        print(
            "\n\033[33mWarning:\033[0m #wrong_elements / #total_elements =",
            wrong_occurences / size,
            "at relative tolerance",
            rtol,
            "!",
        )
        print(
            "\033[33mDont't panic:\033[0m If the above relative number of wrong"
            " elements is very small (e.g. 1e-4), then you can ignore the test"
            " failure."
        )
        return False
    return True


@always_inline
fn is_close(
    arr: Array, arr_torch: PythonObject, rtol: Float32 = 10e-5
) raises -> Bool:
    """
    Asserts that the values in the endia Array and the torch tensor are equal.
    """
    if arr.is_complex():
        var real = real(arr)
        var imag = imag(arr)
        var torch_real = arr_torch.real
        var torch_imag = arr_torch.imag
        return is_close_to_tensor(
            real, torch_real, rtol
        ) and is_close_to_tensor(imag, torch_imag, rtol)
    return is_close_to_tensor(arr, arr_torch, rtol)
