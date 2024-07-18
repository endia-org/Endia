import endia as nd
import endia.nn as nn
import endia.optim as optim

from time import now


def fill_sin_(inout curr: nd.Array, arg: nd.Array):
    for i in range(arg.size()):
        curr.store(i, math.sin(50 * (arg.load(i) + 1) / 2))


def benchmark_mlp_imp():
    print("\nRunning MLP benchmark in eager mode.")

    batch_size = 128
    num_iters = 1000
    every = 1000

    avg_loss = SIMD[dtype, 1](0)
    x = nd.Array(List(batch_size, 1))
    y = nd.Array(List(batch_size, 1))
    mlp = nn.MLP(
        List(1, 32, 64, 128, 128, 128, 64, 32, 1), compute_backward=True
    )
    optimizer = optim.Adam(
        mlp.params(), lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
    )

    fwd_time = Float64(0)
    bwd_time = Float64(0)
    opt_time = Float64(0)
    end_time = Float64(0)

    for i in range(1, num_iters + 1):
        start = now()

        start_init = now()
        nd.randu_(x, min=0, max=1)
        fill_sin_(y, x)
        end_init = now()

        start_fwd = now()
        pred = mlp.forward(x)
        loss = nd.mse(pred, y)
        end_fwd = now()

        if i == 1:
            nd.utils.visualize_graph(loss, "./assets/mlp_imp_graph")

        avg_loss += loss.load(0)

        start_bwd = now()
        loss.backward()
        end_bwd = now()

        start_opt = now()
        optimizer.step()
        end_opt = now()

        zero_grad_time_start = now()
        loss.zero_grad()
        zero_grad_time_end = now()

        end = now()

        fwd_time += (end_fwd - start_fwd) / 1000000000
        bwd_time += (end_bwd - start_bwd) / 1000000000
        opt_time += (end_opt - start_opt) / 1000000000
        end_time += (end - start) / 1000000000

        if i % every == 0:
            print("Iter: ", i, " Loss: ", avg_loss / every)
            avg_loss = 0

            fwd_time /= every
            bwd_time /= every
            opt_time /= every
            end_time /= every

            print(
                "Total:",
                end_time,
                "Fwd: ",
                fwd_time,
                " Bwd: ",
                bwd_time,
                " Optim: ",
                opt_time,
            )

            init_time = 0
            fwd_time = 0
            bwd_time = 0
            opt_time = 0
            zero_grad_time = 0
            end_time = 0
