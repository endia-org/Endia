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

from benchmarks import *


def run_benchmarks():
    # benchmark_foo_grad()

    # benchmark_mlp_imp()
    # benchmark_mlp_func()
    # benchmark_mlp_jit()
    # benchmark_mlp_jit_with_MAX()

    fft_benchmark()
    fft2_benchmark()
    fftn_benchmark()
