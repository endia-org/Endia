# JIT Compilation in Endia (experimental)

Endia provides Just-In-Time (JIT) compilation capabilities to optimize and cache function execution for improved performance. The JIT compiler traces function calls, optimizes subgraphs, and caches the results for future use.

By default the JIT compiler uses a set of home made optimizations to improve performance. However, you can also leverage Modular's **MAX Engine** 🔥. In certain scenarios, using **MAX** can provide additional performance benefits.

## JIT Compilation Process

1. **Tracing**: The function is traced, capturing all operations performed on the input arrays.

2. **Branch Handling**: The JIT compiler can handle functions with conditional statements. It compares operations during execution with previously captured traces, branching when necessary and storing new paths.

3. **Graph Building**: The traced operations form a computation graph, with each node representing an operation or an array.

4. **Subgraph Optimization**: The graph is divided into subgraphs marked by breakpoints. These subgraphs are optimized by fusing elementwise operations and applying other optimizations.

5. **Caching**: Optimized subgraphs are cached for future use. When the same breakpoint is encountered in subsequent executions, the cached, optimized subgraph is used instead of recomputing.

## Performance Considerations

- JIT compilation can significantly improve performance for functions that are called multiple times with similar input shapes.
- The first call to a JIT-compiled function may be slower due to the tracing and optimization process.
- For functions with highly dynamic control flow that changes frequently based on input data, JIT compilation may not provide significant benefits.