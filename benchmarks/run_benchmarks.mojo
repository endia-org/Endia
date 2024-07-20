from benchmarks import *


def run_benchmarks():
    # benchmark_foo_grad()
    benchmark_mlp_imp()
    benchmark_mlp_func()
    benchmark_mlp_jit()
    benchmark_mlp_jit_with_MAX()
