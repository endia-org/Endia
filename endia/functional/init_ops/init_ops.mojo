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

from endia import Array
from endia.utils import dtype
import random

from endia.functional._utils import compute_nd_index, compute_storage_offset

########################################################################
# Initialization Ops
########################################################################


fn array(*dims: Int, requires_grad: Bool = False) raises -> Array:
    var shape = List[Int]()
    for dim in dims:
        shape.append(dim)
    return Array(shape, requires_grad)


fn array(arr_str: StringLiteral, requires_grad: Bool = False) raises -> Array:
    return Array(String(arr_str), requires_grad)


fn arange_(inout arg: Array) raises:
    if arg.is_complex():
        for i in range(arg.size()):
            arg.store_complex(i, i, 0)
    else:
        for i in range(arg.size()):
            arg.store(i, i)


fn arange(shape: List[Int], requires_grad: Bool = False) raises -> Array:
    var res = Array(shape, requires_grad)
    arange_(res)
    return res


fn arange_like(arg: Array) raises -> Array:
    return arange(arg.shape(), arg.requires_grad())


fn arange_complex(
    shape: List[Int], requires_grad: Bool = False
) raises -> Array:
    var real = arange(shape, requires_grad)
    var imag = arange(shape, requires_grad)
    return complex(real, imag, requires_grad)


fn zeros_(inout arg: Array) raises:
    if arg.is_complex():
        memset_zero(arg.data(), arg.base().size() * 2)
    else:
        memset_zero(arg.data(), arg.base().size())


fn zeros(shape: List[Int], requires_grad: Bool = False) raises -> Array:
    var res = Array(shape, requires_grad)
    zeros_(res)
    return res


fn zeros_like(arg: Array) raises -> Array:
    return zeros(arg.shape(), arg.requires_grad())


fn ones_(inout arg: Array) raises:
    if arg.is_complex():
        for i in range(arg.size()):
            arg.store_complex(i, 1, 0)
    else:
        for i in range(arg.size()):
            arg.store(i, 1)


fn ones(shape: List[Int], requires_grad: Bool = False) raises -> Array:
    var res = Array(shape, requires_grad)
    ones_(res)
    return res


fn ones_like(arg: Array) raises -> Array:
    return ones(arg.shape(), arg.requires_grad())


fn eye_(inout arg: Array) raises:
    if arg.shape()[0] != arg.shape()[1] or not arg.ndim() == 2:
        raise "Error: eye_ requires a square matriarg"
    if arg.is_complex():
        for i in range(arg.size()):
            arg.store_complex(i, 1, 0)
    else:
        for i in range(arg.shape()[1]):
            arg.store(i + i * arg.shape()[1], 1)


fn eye(n: Int, requires_grad: Bool = False) raises -> Array:
    var res = Array(List(n, n), requires_grad)
    eye_(res)
    return res


fn eye_like(arg: Array) raises -> Array:
    if arg.shape()[0] != arg.shape()[1] or not arg.ndim() == 2:
        raise "Error: eye_ requires a square matriarg"
    return eye(arg.shape()[0], arg.requires_grad())


fn fill_(inout arg: Array, value: SIMD[dtype, 1]) raises:
    for i in range(arg.size()):
        arg.store(i, value)
    # if arg.is_complex():
    #     for i in range(arg.size()):
    #         arg.store_complex(i, value, 0)
    # else:
    #     for i in range(arg.size()):
    #         arg.store(i, value)


fn full(
    shape: List[Int], value: SIMD[dtype, 1], requires_grad: Bool = False
) raises -> Array:
    var res = Array(shape, requires_grad)
    fill_(res, value)
    return res


fn fill_like(arg: Array, value: SIMD[dtype, 1]) raises -> Array:
    return full(arg.shape(), value, arg.requires_grad())


fn indeces(
    shape: List[Int],
    stride: List[Int],
    storage_offset: Int,
    requires_grad: Bool = False,
) raises -> Array:
    var res = Array(shape, requires_grad)
    var total_elements = 1
    for i in range(shape.size):
        total_elements *= shape[i]
    for index in range(total_elements):
        var nd_index = compute_nd_index(index, shape)
        var storage_offset = compute_storage_offset(
            nd_index, stride, storage_offset
        )
        res.store[1](index, storage_offset)
    return res


fn randu_(inout arg: Array, min: Float64 = 0, max: Float64 = 1) raises:
    random.seed()
    var data = arg.data()
    var size = arg.base().size()
    random.rand(data, size)
    for i in range(size):
        data[i] = min + data[i] * (max - min)


fn randu(
    shape: List[Int],
    min: Float64 = 0,
    max: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var res = Array(shape, requires_grad)
    randu_(res, min, max)
    return res


fn randu_like(
    inout arg: Array, min: Float64 = 0, max: Float64 = 1
) raises -> Array:
    return randu(arg.shape(), min, max, arg.requires_grad())


fn randn_(inout arg: Array, mean: Float64 = 0, std: Float64 = 1) raises:
    random.seed()
    var data = arg.data()
    var size = arg.base().size()
    random.randn(data, size, mean, std)


fn randn(
    shape: List[Int],
    mean: Float64 = 0,
    std: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var res = Array(shape, requires_grad)
    randn_(res, mean, std)
    return res


fn randn_like(
    inout arg: Array, mean: Float64 = 0, std: Float64 = 1
) raises -> Array:
    return randn(arg.shape(), mean, std, arg.requires_grad())


fn rand_he_normal_(inout arg: Array, fan_in: Float64 = 1) raises:
    var std = math.sqrt(2 / fan_in)
    randn_(arg, 0, std)


fn rand_he_normal(
    shape: List[Int], fan_in: Float64 = 1, requires_grad: Bool = False
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_he_normal_(res, fan_in)
    return res


fn rand_he_normal_like(arg: Array, fan_in: Float64 = 1) raises -> Array:
    return rand_he_normal(arg.shape(), fan_in, arg.requires_grad())


fn rand_he_uniform_(inout arg: Array, fan_in: Float64 = 1) raises:
    var limit = math.sqrt(6.0 / fan_in)
    randu_(arg, -limit, limit)


fn rand_he_uniform(
    shape: List[Int], fan_in: Float64 = 1, requires_grad: Bool = False
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_he_uniform_(res, fan_in)
    return res


fn rand_he_uniform_like(arg: Array, fan_in: Float64 = 1) raises -> Array:
    return rand_he_uniform(arg.shape(), fan_in, arg.requires_grad())


fn rand_xavier_normal_(
    inout arg: Array, fan_in: Float64 = 1, fan_out: Float64 = 1
) raises:
    var std = math.sqrt(2.0 / (fan_in + fan_out))
    randn_(arg, 0, std)


fn rand_xavier_normal(
    shape: List[Int],
    fan_in: Float64 = 1,
    fan_out: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_xavier_normal_(res, fan_in, fan_out)
    return res


fn rand_xavier_normal_like(
    inout arg: Array, fan_in: Float64 = 1, fan_out: Float64 = 1
) raises -> Array:
    return rand_xavier_normal(arg.shape(), fan_in, fan_out, arg.requires_grad())


fn rand_xavier_uniform_(
    inout arg: Array, fan_in: Float64 = 1, fan_out: Float64 = 1
) raises:
    var limit = math.sqrt(6.0 / (fan_in + fan_out))
    randu_(arg, -limit, limit)


fn rand_xavier_uniform(
    shape: List[Int],
    fan_in: Float64 = 1,
    fan_out: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_xavier_uniform_(res, fan_in, fan_out)
    return res


fn rand_xavier_uniform_like(
    inout arg: Array, fan_in: Float64 = 1, fan_out: Float64 = 1
) raises -> Array:
    return rand_xavier_uniform(
        arg.shape(), fan_in, fan_out, arg.requires_grad()
    )


fn rand_lecun_normal_(inout arg: Array, fan_in: Float64 = 1) raises:
    var std = math.sqrt(1.0 / fan_in)
    randn_(arg, 0, std)


fn rand_lecun_normal(
    shape: List[Int], fan_in: Float64 = 1, requires_grad: Bool = False
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_lecun_normal_(res, fan_in)
    return res


fn rand_lecun_normal_like(
    inout arg: Array, fan_in: Float64 = 1
) raises -> Array:
    return rand_lecun_normal(arg.shape(), fan_in, arg.requires_grad())


fn rand_lecun_uniform_(inout arg: Array, fan_in: Float64 = 1) raises:
    var limit = math.sqrt(3.0 / fan_in)
    randu_(arg, -limit, limit)


fn rand_lecun_uniform(
    shape: List[Int], fan_in: Float64 = 1, requires_grad: Bool = False
) raises -> Array:
    var res = Array(shape, requires_grad)
    rand_lecun_uniform_(res, fan_in)
    return res


fn rand_lecun_uniform_like(
    inout arg: Array, fan_in: Float64 = 1
) raises -> Array:
    return rand_lecun_uniform(arg.shape(), fan_in, arg.requires_grad())


fn complex(
    real: Array, imag: Array, requires_grad: Bool = False
) raises -> Array:
    # compare shapes, they must be equal
    if real.ndim() != imag.ndim():
        raise "Error: real and imag parts must have the same shape"
    for i in range(real.ndim()):
        if real.shape()[i] != imag.shape()[i]:
            raise "Error: real and imag parts must have the same shape"
    var res = Array(real.shape(), requires_grad, True)
    for i in range(res.size()):
        res.store_complex(i, real.load(i), imag.load(i))
    return res


fn randn_complex(
    shape: List[Int],
    mean: Float64 = 0,
    std: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var real = randn(shape, mean, std, requires_grad)
    var imag = randn(shape, mean, std, requires_grad)
    return complex(real, imag, requires_grad)


fn randu_complex(
    shape: List[Int],
    min: Float64 = 0,
    max: Float64 = 1,
    requires_grad: Bool = False,
) raises -> Array:
    var real = randu(shape, min, max, requires_grad)
    var imag = randu(shape, min, max, requires_grad)
    return complex(real, imag, requires_grad)


fn fill_complex(
    shape: List[Int],
    value_real: Float64,
    value_imag: Float64,
    requires_grad: Bool = False,
) raises -> Array:
    var real = full(shape, value_real, requires_grad)
    var imag = full(shape, value_imag, requires_grad)
    return complex(real, imag, requires_grad)
