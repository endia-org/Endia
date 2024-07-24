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
from endia.utils import (
    list_to_array_shape,
    array_shape_to_list,
    setup_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)


####--------------------------------------------------------------------------------------------------------------------####
# Concatenate
####--------------------------------------------------------------------------------------------------------------------####


fn concat_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
    """
    Computes the shape of an array after concatenation.

    Args:
        curr: The ArrayShape to store the result of the computation.
        args: The ArrayShapes to concatenate, and the axis to concatenate along encoded in an ArrayShape.
    """
    var num_args = len(args) - 1
    var axis = array_shape_to_list(args[num_args])
    var arg0 = args[0]
    if axis[0] < 0:
        axis[0] = arg0.ndim() + axis[0]
    if axis[0] >= arg0.ndim():
        raise "Error: Axis out of bounds"
    if axis.size != 1:
        raise "Error: Only one axis is allowed for concatenation"
    for i in range(1, num_args):
        var arg = args[i]
        if arg0.ndim() != arg.ndim():
            raise "Error: Incompatible shapes for concatenation"
        for j in range(arg0.ndim()):
            if arg0.shape()[j] != arg.shape()[j]:
                raise "Error: Incompatible shapes for concatenation"
    var shape = arg0.shape()
    shape[axis[0]] *= num_args
    curr.setup(shape)


fn concat_fwd(inout curr: Array, args: List[Array]) raises:
    """
    Performs the forward pass for the concat operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

    Args:
        curr: The current array to store the result (modified in-place).
        args: The arrays to concatenate.

    #### Note:
    The information of the shape computation is stored in the ArrayShape object of the curr array.
    """
    var arg_shapes = curr.array_shape().args()
    var axis = array_shape_to_list(arg_shapes[len(arg_shapes) - 1])[0]
    setup_shape_and_data(curr)
    if axis < 0:
        axis = curr.ndim() + axis
    # print(axis)
    # let's slice into curr and populate with the correct values in the corresponding arg


fn concat_vjp(
    primals: List[Array], grad: Array, out: Array
) raises -> List[Array]:
    """
    Computes the vector-Jacobian product for the concat operation.

    Args:
        primals: A list containing the primal input arrays.
        grad: The gradient of the output with respect to some scalar function.
        out: The output of the forward pass.

    Returns:
        A list containing the gradients with respect to the input arrays.

    #### Note:
    The vector-Jacobian product for concat is computed by returning an empty list.
    """
    var primals_grads = List[Array]()
    return primals_grads


fn concat(args: List[Array], axis: Int) raises -> Array:
    """
    Concatenates the input arrays along the given axis.

    Args:
        args: The arrays to concatenate.
        axis: The axis along which to concatenate.

    Returns:
        The concatenated array.
    """
    var arg_shapes = List[ArrayShape]()
    for arg in args:
        arg_shapes.append(arg[].array_shape())
    arg_shapes.append(list_to_array_shape(axis))
    var arr_shape = setup_array_shape(arg_shapes, "concat", concat_shape)

    return op_array(
        arr_shape, args, NA, "concat", concat_fwd, default_jvp, concat_vjp, True
    )
