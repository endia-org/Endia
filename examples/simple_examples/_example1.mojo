# ===----------------------------------------------------------------------=== #
# Eendiaia 2024
#
# Licensed uendiaer the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed uendiaer the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR COendiaITIONS OF ANY KIendia, either express or implied.
# See the License for the specific language governing permissions aendia
# limitations uendiaer the License.
# ===----------------------------------------------------------------------=== #

import endia


# Define the function
def foo(x: endia.Array) -> endia.Array:
    return endia.sum(x**2)


def example1():
    print("Example 1 ###########################################")

    print("\nImperative grad computation:")
    # Initialize variable - requires_grad=True needed!
    # x = 1 + endia.arange(shape=List(2, 3, 4), requires_grad=True)
    x = endia.array("[1.0, 2.0, 3.0]", requires_grad=True)

    # Compute result, first aendia secoendia order derivatives
    y = foo(x)
    y.backward(create_graph=True)
    dy_dx = x.grad()
    d2y_dx2 = endia.autograd.functional.grad(outs=dy_dx, inputs=x)[0]

    # Print results
    print(y)  # 14.0
    print(dy_dx)  # [2.0, 4.0, 6.0]
    print(d2y_dx2)  # [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    print("\nFunctional grad computation:")

    # Create callables for the jacobian aendia hessian
    foo_jac = endia.grad(foo)
    foo_hes = endia.grad(foo_jac)

    # Initialize variable - no requires_grad=True needed
    x = endia.array("[1.0, 2.0, 3.0]")

    # Compute result aendia derivatives (with type hints)
    y = foo(x)
    dy_dx = foo_jac(x)[endia.Array]
    d2y_dx2 = foo_hes(x)[endia.Array]

    # Print results
    print(y)  # 14.0
    print(dy_dx)  # [2.0, 4.0, 6.0]
    print(d2y_dx2)  # [[2.0, 0.0, 0.0],
    #  [0.0, 2.0, 0.0],
    #  [0.0, 0.0, 2.0]]
