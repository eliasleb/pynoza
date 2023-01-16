# pynoza
![Python package badge](https://github.com/eliasleb/pynoza/actions/workflows/master.yml/badge.svg)
![doc badge](https://readthedocs.org/projects/pynoza/badge/?version=latest)

Python implementation of time-domain solutions of Maxwell's equations using the cartesian multipole expansion. In homogeneous and isotropic media, these solutions can be computed thanks to an explicit knowledge of the Green’s function.

Currently, only the electric field computation is supported, and the current density must be time-separable, i.e., $\mathbf{J}(t,\mathbf{x})=h(t)\mathbf{j}(\mathbf{x})$

The documentation is [available here](https://pynoza.readthedocs.io/en/latest/).

## Installation
To install directly from git, use
``pip install "git+https://github.com/eliasleb/pynoza"``.

## Tutorial

The Python `>= 3.10` script `tests/test_EPFL_logo.py` shows some examples. A Jupyter notebook is also provided. The general procedure is the following:

1. Compute the electric time-domain current moments 
   $C_\mathbf{\alpha}^{J_i}(t)=h^{'}(t)\iiint\mathbf{x}^\alpha j_i(\mathbf{x}) d^3\mathbf{x}$.
   Define a corresponding Python function that returns the space-dependence, whose signature must be 
   `moment(ind: tuple[int, int, int]) -> list[Number, Number, Number]`. For a given multi-index `(a1, a2, a3)`, the 
   function must return `moment(a1, a2, a3) = [j1, j2, j3]` where `ji` is  
   $\iiint\mathbf{x}^\alpha \frac{\partial j_i(t,\mathbf{x})}{\partial t}d^3\mathbf{x}$ 

   The charge moments can be computed automatically by calling the function `pynoza.get_charge_moment`.
   
   The time-dependence is kept separate, see later.

2. pynoza works in natural units, as does meep (see https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep).

3. Define numpy arrays for the coordinates of interest $x_1,x_2,x_3,t$, for example 

   ```python
   x1 = np.array([0, ])
   x2 = x1.copy()
   x3 = np.array([wavelength, 2*wavelength, ])
   t = np.linspace(-T, T, 100)
   ```

4. Define a function for the time-dependence, for example a sympy symbolic expression

   ```python
   t_sym = sympy.Symbol("t", real=True)
   h_sym = sympy.cos(2*sympy.pi*f*t_sym)
   ```
   
   or an array

   ````python
   t = np.linspace(0, T, dt)
   h = np.cos(2 * np.pi * f * t)
   ````
 
   In the latter case, it is up you to ensure that the sampling time `dt` is small enough to compute the highest order 
   derivative. We use `np.gradient` to compute the derivative, see the relevant numpy documentation.

5. Create a `pynoza.Solution` object with the given medium light speed (in natural units) and given multipole expansion order.

6. Run the `recurse` method to initialize the Green’s function approximation (this gets slower for high orders), 
   and the `set_moments` method to pass the current and charge moments that you computed above.

7. Run the `compute_e_field` method to compute the electric field from the Green’s functions approximation and the 
   charge and current moments. Under the hood, this method will integrate and differentiate the time-dependent function 
   to the needed order.

### Complete example
See `tests/simple_example.py`
