# Fast Fourier Transform

The FFT module in Endia is a collection of algorithms and utilities for performing (inverse) Fast Fourier Transforms. The module is designed to be very similar to the *torch.fft* module, with the benefit of being written in pure Mojo in only a few lines of code. Our benchmarks show that Endia's fft module performs on par with the PyTorch implementation: Depending on the shape of the input signals (all transformed dimensions must be  power of 2 for now), you can expet a perfomance gain in the range of [-80%, 50%].

<div align="center">
  <img src="../../../assets/fft_title_image.png" alt="Endia Stack concept Image" /> <!-- style="max-width: 800px;" -->
</div>

## Overview

TODO