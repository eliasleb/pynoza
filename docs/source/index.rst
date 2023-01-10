pynoza
======
Pynoza is a Python implementation of the time-domain multipole expansion of electromagnetic fields in Cartesian
coordinates.

Currently, only the electric field computation is supported, and the current density must be time-separable, i.e.,
the current density must be the product of two time- and space-dependent functions.

The source is code is available on `GitHub <https://github.com/eliasleb/pynoza>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autoclass:: pynoza.solution.Solution
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pynoza.helpers.get_charge_moment
