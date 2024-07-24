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


def viz_example1():
    a = nd.arange(List(2, 3), requires_grad=True)
    b = nd.arange(List(3, 4), requires_grad=True)
    c = nd.arange(List(2, 2, 4), requires_grad=True)

    res = nd.sum(a @ b + c)

    nd.utils.visualize_graph(res, "./assets/example1_graph")
