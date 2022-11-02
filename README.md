# pynoza
![Python package badge](https://github.com/eliasleb/pynoza/actions/workflows/master.yml/badge.svg)

Python implementation of time-domain solutions of Maxwell's equations using the cartesian multipole expansion. In homogeneous and isotropic media, these solutions can be computed thanks to an explicit knowledge of the Green’s function.

Currently, only the electric field computation is supported, and the current density must be time-separable, i.e., $\mathbf{J}(t,\mathbf{x})=h(t)\mathbf{j}(\mathbf{x})$

## Installation
To install, clone the repository using `git clone https://c4science.ch/source/pynoza.git`.

Then, from the root `pynoza` folder, run `pip install .`.

## Tutorial

The Python `>= 3.10` script `tests/test_EPFL_logo.py` shows some examples. A Jupyter notebook is also provided. The general procedure is the following:

1. Given the current density $\mathbf{J}(t,\mathbf{x})$, compute the charge density $\rho$, given by $\frac{\partial\rho(t,\mathbf{x})}{\partial t}=-\nabla\cdot\mathbf{J}(t,\mathbf{x})$.

2. Compute the electric time-domain current moments $C_\mathbf{\alpha}^{J_i}(t)=h^{'}(t)\iiint\mathbf{x}^\alpha \frac{\partial j_i(t,\mathbf{x})}{\partial t}d^3\mathbf{x}$ and the electric time-domain charge moment $C_\mathbf{\alpha}^{\rho_i}(t)=\partial^{-1}h(t)\iiint\mathbf{x}^\alpha \frac{\partial \rho(\mathbf{x})}{\partial x_i}d^3\mathbf{x}$. Define two corresponding Python functions that return the space-dependence, whose signature must be `moment(ind: tuple[int, int, int]) -> list[Number, Number, Number]`. The time-dependence is kept separate, see later.

3. pynoza works in natural units, as does meep (see https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep).

4. Define numpy arrays for the coordinates of interest $x_1,x_2,x_3,t$, for example 

   ```python
   x1 = np.array([0, ])
   x2 = x1.copy()
   x3 = np.array([wavelength, 2*wavelength, ])
   t = np.linspace(-T, T, 100)
   ```

5. Define a symbolic function for the time-dependence, for example

   ```python
   t_sym = sympy.Symbol("t", real=True)
   h_sym = sympy.cos(2*sympy.pi*f*t_sym)
   ```

6. Create a `pynoza.Solution` object with the given medium light speed (in natural units) and given multipole expansion order.

7. Run the `recurse` method to initialize the Green’s function approximation (this gets slower for high orders), and the `set_moments` method to pass the current and charge moments that you computed above.

8.  Run the `compute_e_field` method to compute the electric field from the Green’s functions approximation and the charge and current moments. Under the hood, this method will integrate and differentiate the time-dependent function `h_sym` to the needed order.
