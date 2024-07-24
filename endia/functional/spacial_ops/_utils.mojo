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

# from endia import Array


# trait DifferentiableSpacialOp:
#     @staticmethod
#     fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
#         ...

#     @staticmethod
#     fn fwd(arg0: Array, axis: List[Int]) raises -> Array:
#         ...

#     @staticmethod
#     fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
#         ...

#     @staticmethod
#     fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
#         ...

#     @staticmethod
#     fn __call__(inout curr: Array, args: List[Array]) raises:
#         ...
