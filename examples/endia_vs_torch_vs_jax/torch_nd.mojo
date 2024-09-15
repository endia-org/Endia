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


from endia import Tensor, sum, arange
from endia.autograd import grad
import endia.autograd.functional as F


def foo(x: Tensor) -> Tensor:
    return sum(x**2)


def example_torch_like():
    x = arange(1.0, 4.0, requires_grad=True)

    y = foo(x)
    dy_dx = grad(outs=y, inputs=x)[0]
    d2y_dx2 = F.hessian(foo, x)

    print(str(y))
    print(str(dy_dx))
    print(str(d2y_dx2))
