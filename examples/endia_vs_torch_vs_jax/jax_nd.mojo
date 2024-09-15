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


from endia import grad, jacobian
from endia import sum, arange, ndarray


def foo(x: ndarray) -> ndarray:
    return sum(x**2)


def example_jax_like():
    # create Callables
    foo_jac = grad(foo)
    foo_hes = jacobian(foo_jac)

    x = arange(1.0, 4.0)

    print(str(foo(x)))
    print(str(foo_jac(x)[ndarray]))
    print(str(foo_hes(x)[ndarray]))
