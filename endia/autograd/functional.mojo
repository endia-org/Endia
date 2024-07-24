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
from endia.compile import Callable
from endia.utils import top_order_rec
from utils import Variant


fn grad(
    f: Callable,
    argnums: List[Int] = List(-1),
) raises -> Callable:
    """
    Take in a Callable and return a Callable that computes the derivative of the out of the
    function/callable wrt. all args.
    """
    var existing_argnums = f.argnums
    existing_argnums.append(argnums)
    return Callable(
        f.func,
        existing_argnums,
        f.order_of_differentiation + 1,
        False,
        False,
    )


fn grad(
    f: fn (List[Array]) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    return Callable(f, argnums, 1, False, False)


fn grad(
    f: fn (Array) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    return Callable(f, argnums, 1, False, False)


fn value_and_grad(
    arg: Variant[Callable, fn (List[Array]) raises -> Array],
    argnums: List[Int] = List(-1),
) raises -> Callable:
    """
    Take in a Callable and return a Callable that computes the value and the derivative of the out of the
    function/callable wrt. all args.
    """
    var a = arg
    if arg.isa[Callable]():
        var _a = a.unsafe_take[Callable]()
        var existing_argnums = _a.argnums
        existing_argnums.append(argnums)
        return Callable(
            _a.func,
            existing_argnums,
            _a.order_of_differentiation + 1,
            False,
            True,
        )
    else:
        var _f = a.unsafe_take[fn (List[Array]) raises -> Array]()
        return Callable(_f, argnums, 1, False, True)


###########################################################################
# The follwoing code is only in this file because function overloadings
# (grad) are seeminly not supported for functions living in differnt files.
# Will move this code to imperative.mojo as soon as possible.
###########################################################################
fn backward(arg: Array, retain_graph: Bool) raises:
    var out = arg

    reset_node_id_recursive(out)
    var trace = List[Array]()
    top_order_rec(out, trace)

    var last_grad = ones(arg.shape())
    # last_grad.requires_grad_(True)
    out.grad_(last_grad)

    # var breakpoint_nodes = List[Array]()

    for i in range(len(trace) - 1, -1, -1):
        var curr = trace[i]
        # print("computing backward:", curr.name())
        var primals = curr.args()

        if primals.size == 0:
            continue

        var vjp = curr.vjp()
        var grad = curr.grad()
        var primals_grads = vjp(primals, grad, curr)

        for j in range(len(primals)):
            var primal = primals[j]
            if primal.requires_grad():
                var primal_grad = primals_grads[j]
                if primal.has_grad():
                    primal_grad = add(primal_grad, primal.grad())
                primal.grad_(primal_grad)

                if primal.has_fxgraph():
                    if primal.is_breakpoint():
                        primal_grad.postpone_as_grpah_output()
                        # if primal.args().size > 0:
                        #     # _ = primal.grad().item(0)
                        #     var primal_grad = primal.grad()
                        #     primal_grad.postpone_as_grpah_output()
                        # else:
                        #     breakpoint_nodes.append(primal)

                if not retain_graph:
                    var primal_grad = primal.grad()
                    primal_grad.clear_args()

    # for breakpoint_node in breakpoint_nodes:
    #     var graph = breakpoint_node[].grad().graph()
    #     var id_in_graph = breakpoint_node[].grad().id_in_graph()
    #     var graph_node = graph[].trace[id_in_graph]
    #     if not graph_node.is_computed:
    #         print("here")
    #         _ = breakpoint_node[].grad().item(0)

    #         print("done")

    # _ = out.item(0) # call this on curr to compute all postponed outeputs of the potentially corresponding fx graph, this will cimpute all outputs at once

    reset_node_id_recursive(out)


fn grad(
    outs: List[Array],
    inputs: List[Array],
    retain_grads: Bool = True,
    retain_graph: Bool = False,
) raises -> Variant[Array, List[Array]]:
    """
    Compute the gradient of outs wrt. inputs.
    """
    for i in range(len(outs)):
        var out = outs[i]
        remove_grad_rec(out)
    for i in range(len(inputs)):
        var input = inputs[i]
        remove_grad_rec(input)
    for i in range(len(outs)):
        var out = outs[i]
        out.backward(retain_graph=retain_graph)
    var final_outs = List[Array]()
    for i in range(len(inputs)):
        var input = inputs[i]
        var gradient = input.grad()
        if not retain_graph:
            gradient.clear_args()
            gradient.remove_grad()
        if not retain_grads:
            input.remove_grad()
        final_outs.append(gradient)

    if len(final_outs) == 1:
        return final_outs[0]
    else:
        return final_outs
