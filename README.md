<div align="center">
  <img src="./assets/titleimage3.png" alt="Title Image" />
</div>

###

**Endia** is a dynamic Array library for Accelerated Scientific Computing. It offers:

- **Automatic differentiation**: Compute derivatives of arbitrary order.
- **Complex number support:** Use Endia for advanced scientific applications.
- **Dual API:** Choose between a PyTorch-like imperative or a JAX-like functional interface.
- **JIT Compilation:** Leverage MAX to speed up training and inference.

<div align="center">
  
  [Website] | [Docs] | [Getting Started] | [Our Mission]

  [Website]: https://endia.org
  [Docs]: https://endia.org
  [Getting Started]: #getting-started
  [Our Mission]: #our-mission

</div>

## Installation

1. **Install [Mojo and MAX](https://docs.modular.com/max/install)** 🔥 (v24.4)

2. **Clone the repository**:

    ```bash
    git clone https://github.com/endia-org/Endia.git
    cd Endia
    ```

3. **Set Up Environment**:

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    Required dependencies: `torch`, `numpy`, `graphviz`. These will be installed automatically by the setup script.

## Getting Started

In this guide, we'll demonstrate how to compute the **value**, **gradient**, and the **Hessian** (i.e. the second-order derivative) of a simple function. First by using Endia's Pytorch-like API and then by using a more Jax-like functional API. In both examples, we initially define a function **foo** that takes an array and returns the sum of the squares of its elements.

### The **Pytorch** way

<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://pytorch.org/docs/stable/index.html">
    <img src="assets/pytorch_logo.png" alt="Endia Logo" width="40">
  </a>
</p>

When using Endia's imperative (PyTorch-like) interface, we compute the gradient of a function by calling the **backward** method on the function's output. This imperative style requires explicit management of the computational graph, including setting `requires_grad=True` for the input arrays (i.e. leaf nodes) and using `retain_graph=True` in the backward method when computing higher-order derivatives.

```python
import endia as nd 

# Define the function
def foo(x: nd.Array) -> nd.Array:
    return nd.sum(x ** 2)

# Initialize variable - requires_grad=True needed!
x = nd.array('[1.0, 2.0, 3.0]', requires_grad=True)

# Compute result, first and second order derivatives
y = foo(x)
y.backward(retain_graph=True)            
dy_dx = x.grad()
d2y_dx2 = nd.grad(outs=dy_dx, inputs=x)[nd.Array]

# Print results
print(y)        # out: [14.0]
print(dy_dx)    # out: [2.0, 4.0, 6.0]
print(d2y_dx2)  # out: [2.0, 2.0, 2.0]
```

### The **JAX** way

<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://jax.readthedocs.io/en/latest/quickstart.html">
    <img src="assets/jax_logo.png" alt="Endia Logo" width="65">
  </a>
</p>

When using Endia's functional (JAX-like) interface, the computational graph is handled implicitly. By calling the `grad` function on foo, we create a `Callable` which computes the gradient. This `Callable` can be passed to the `grad` function again to compute higher-order derivatives.

```python
import endia as nd 

# Define the function
def foo(x: nd.Array) -> nd.Array:
    return nd.sum(x ** 2)

# Create callables for the jacobian and hessian
foo_jac = nd.grad(foo)
foo_hes = nd.grad(foo_jac)

# Initialize variable - no requires_grad=True needed
x = nd.array('[1.0, 2.0, 3.0]')

# Compute result and derivatives
y = foo(x)
dy_dx = foo_jac(x)[nd.Array]
dy2_dx2 = foo_hes(x)[nd.Array]

# Print results
print(y)        # out: [14.0]
print(dy_dx)    # out: [2.0, 4.0, 6.0]
print(dy2_dx2)  # out: [2.0, 2.0, 2.0]
```

*And there is so much more! Endia can handle complex valued functions, can perform both forward and reverse-mode automatic differentiation, it even has a builtin JIT compiler to make things go brrr. Explore the full **list of features** in the [documentation](https://endia.org).*

## Our Mission

- 🧠 **Advance AI & Scientific Computing:** Push boundaries with clear and understandable algorithms
- 🚀 **Mojo-Powered Clarity:** High-performance open-source code that remains readable and pythonic through and through
- 📐 **Explainability:** Prioritize clarity and educational value over exhaustive features

<!-- This project originated from a [Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/quickstart) endeavor, aiming to build a tool that would allow people to more easily explore the inner workings of AI models, giving them the power to find, tweak, and reason about internal circuits and learned features. This field of research is in its infancy, and it often requires knowledge about the entire stack of machine learning tools. People who deeply understand PyTorch and are able to conduct high-level research with it simultaneously are actually quite rare, and we need more of them! The goal of Endia is to make this knowledge more accessible and to provide a stepping stone for people to dive deeper into the field. As of now, Endia is in its early stages, and we are working on expanding its feature set and making it more robust. We are also developing a series of tutorials and educational materials to help **you** get started with Endia and to understand the underlying concepts of automatic differentiation, complex numbers, optimization, and more. -->

## Contributing

Contributions to Endia are welcome! If you'd like to contribute, please follow the contribution guidelines in the [CONTRIBUTING.md](https://github.com/endia-org/Endia/blob/main/CONTRIBUTING.md) file in the repository.

## Citation

If you use Endia in your research or project, please cite it as follows:

```bibtex
@software{Fehrenbach_Endia_-_Scientific_2024,
author = {Fehrenbach, Tillmann},
license = {Apache-2.0},
month = jul,
title = {{Endia - Scientific Computing in Mojo}},
url = {https://github.com/endia-org/Endia},
version = {24.4.0},
year = {2024}
}
```

## License

Endia is licensed under the [Apache-2.0 license](https://github.com/endia-org/Endia?tab=Apache-2.0-1-ov-file).
