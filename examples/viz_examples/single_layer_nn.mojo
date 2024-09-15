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


import endia as nd


def single_layer_viz_example():
    # initialization
    weight1 = nd.randu(List(3, 4), requires_grad=True)
    bias1 = nd.randu(List(4), requires_grad=True)
    weight2 = nd.randu(List(4, 5), requires_grad=True)
    bias2 = nd.randu(List(5), requires_grad=True)

    # forward
    input = nd.ones(List(2, 3))
    hidden = nd.relu(input @ weight1 + bias1)
    pred = hidden @ weight2 + bias2

    # visualize
    nd.utils.visualize_graph(pred, "./single_layer_nn")
