# Endia - Changelog

## Branch: Nightly, Version: 24.4.1.

### **New**

- Adding more primitive operations to the MAX graph utils.
- Added a JIT mlp benchmark without MAX in mlp_jit.mojo. Renamed the file with the JIT with MAX mlp benchmark to mlp_jit_with_MAX.mojo.
- Adding spacial operations: conv1d, conv2d, conv3d, max_pool1d, max_pool2d, max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d. No MAX conversion yet. (TODO!)
- Adding the corresponding tests for the spacial operations. No edge case testing yet. (TODO!)

### **Changed**

- Changed atol ro rtol in close_to function in utils module. This function is used to compare an Endia Array with a PyTorch Tensor. Using atol resulted in a lot of test failures, using rtol makes a bit more sense here, since small numerical errors are expected.
- Changed the vjp and jvp function of the relu acitvation function to use the greater_equal method instead of the ge_zeo method.
- In benchmarks, changed the loss and timing output to not average over the first til the last iteration, but show these values for every 500th interation. It seemed that the JIT version with MAX was consistently showing smaller loss, but with this new, more fine-grained output, it is clear that this is not the case, which is a releave.
- Changed the dim sizes in most tests from values smaller than the minimum simd width value (possibly 8 or 16) to values larger than the minimum simd width value. All adapted test pass as well.

### **Fixed**