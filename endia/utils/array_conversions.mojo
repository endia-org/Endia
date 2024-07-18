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
    arr: Array, arr_torch: PythonObject, atol: Float32 = 10e-3
) raises -> Bool:
    """
    Asserts that the values in the endia Array and the torch tensor are equal.
    """
    # var int_tol = int(100000)
    var tolerance = int(1 / atol)
    var shape = arr.shape()
    var size = 1
    for i in range(shape.size):
        size *= shape[i]
    var flattened = arr_torch.flatten()
    var rel_false_values = SIMD[dtype, 1](0)
    for i in range(size):
        var arr_val = (arr.load(i) * tolerance).roundeven()
        var torch_val = (
            flattened[i].to_float64().cast[dtype]() * tolerance
        ).roundeven()
        if not isnan(arr_val) and not isinf(arr_val):
            if arr_val != torch_val:
                # if the read in number is greater than a certain threshold, neglect the values after the comma
                var int_tol = SIMD[dtype, 1](100)
                if (
                    abs(arr.load(i)) > int_tol
                    and abs(flattened[i].to_float64().cast[dtype]()) > int_tol
                ):
                    if (
                        abs(arr_val // tolerance).roundeven()
                        == abs(torch_val // tolerance).roundeven()
                    ):
                        continue
                print(
                    "Incorrect value at index",
                    i,
                    " - endia_val=",
                    arr.load(i),
                    " - torch_val=",
                    flattened[i].to_float64().cast[dtype](),
                )
                rel_false_values += 1
                return False
    return True


@always_inline
fn is_close(
    arr: Array, arr_torch: PythonObject, atol: Float32 = 10e-3
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
            real, torch_real, atol
        ) and is_close_to_tensor(imag, torch_imag, atol)
    return is_close_to_tensor(arr, arr_torch, atol)
