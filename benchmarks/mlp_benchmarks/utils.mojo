alias BENCHMARK_FUNC = def () -> (Float32,Float64,Float64,Float64)
alias BENCHMARK_FUNC_2 = def () -> (Float32,Float64,Float64,Float64,Float64)

def benchmark_avg(rounds:Int,benchmark_func:BENCHMARK_FUNC,func_text:String):
    print("\nRunning MLP benchmark",func_text,rounds,"times:")

    loss = Float32(0)
    time_all = Float64(0)
    time_fwd = Float64(0)
    time_optim = Float64(0)

    loss_total = Float32(0)
    time_all_total = Float64(0)
    time_fwd_total = Float64(0)
    time_optim_total = Float64(0)

    for _ in range(rounds):
        loss,time_all,time_fwd,time_optim = benchmark_func()

        loss_total += loss
        time_all_total += time_all
        time_fwd_total += time_fwd
        time_optim_total += time_optim

    # print loss
    print("Iter: ", rounds,"x",1000, " Avg Loss: ", loss_total/rounds)
    
    print(
        "Avg Total: ",
        time_all_total/rounds,
        " Avg Value_and_Grad: ",
        time_fwd_total/rounds,
        " Avg Optim: ",
        time_optim_total/rounds,
    )
   

def benchmark_avg_2(rounds:Int,benchmark_func:BENCHMARK_FUNC_2,func_text:String):
    print("\nRunning MLP benchmark",func_text,rounds,"times:")

    loss = Float32(0)
    time_all = Float64(0)
    time_fwd = Float64(0)
    time_bwd = Float64(0)
    time_optim = Float64(0)

    loss_total = Float32(0)
    time_all_total = Float64(0)
    time_fwd_total = Float64(0)
    time_bwd_total = Float64(0)
    time_optim_total = Float64(0)

    for _ in range(rounds):
        loss,time_all,time_fwd,time_bwd,time_optim = benchmark_func()

        loss_total += loss
        time_all_total += time_all
        time_fwd_total += time_fwd
        time_bwd_total += time_bwd
        time_optim_total += time_optim

    # print loss
    print("Iter: ", rounds,"x",1000, " Avg Loss: ", loss_total/rounds)
    
    print(
        "Avg Total: ",
        time_all_total/rounds,
        " Avg Fwd: ",
        time_fwd_total/rounds,
        " Avg Bwd: ",
        time_bwd_total/rounds,
        " Avg Optim: ",
        time_optim_total/rounds,
    )
   
