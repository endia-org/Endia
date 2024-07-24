# ===----------------------------------------------------------------------=== #
# Endia 2024
# 
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# from extensibility import Tensor, empty_tensor
from max.tensor import Tensor
import python
from endia import Array


@always_inline
fn memcpy_to_numpy(array: PythonObject, tensor: Array) raises:
    var dst = DTypePointer[dtype](
        address=int(array.__array_interface__["data"][0])
    )
    var src = tensor.data()
    var length = tensor.size()
    memcpy(dst, src, length)


@always_inline
fn shape_to_python_list(shape: List[Int]) raises -> PythonObject:
    var python_list = python.Python.evaluate("list()")
    for i in range(len(shape)):
        _ = python_list.append(shape[i])
    return python_list^


@always_inline
fn get_np_dtype[dtype: DType](np: PythonObject) raises -> PythonObject:
    @parameter
    if dtype.is_float32():
        return np.float32
    elif dtype.is_int32():
        return np.int32
    elif dtype.is_int64():
        return np.int64
    elif dtype.is_uint8():
        return np.uint8

    raise "Unknown datatype"


@always_inline
fn array_to_numpy(tensor: Array, np: PythonObject) raises -> PythonObject:
    var shape = shape_to_python_list(tensor.shape())
    var tensor_as_numpy = np.zeros(shape, np.float32)
    _ = shape^
    memcpy_to_numpy(tensor_as_numpy, tensor)
    return tensor_as_numpy^


fn tensor_to_array(src: Tensor[dtype]) raises -> Array:
    var shape = List[Int]()
    for i in range(src.rank()):
        shape.append(src.shape()[i])
    var dst = Array(shape)
    var dst_data = dst.data()
    var src_data = src._ptr
    memcpy(dst_data, src_data, dst.size())
    return dst
