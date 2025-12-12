Coordinate Conversion Functions
===============================
This functions are used to convert between different coordinate systems (e.g. from cartesian to polar, from polar to
cartesian, etc.). The functions are implemented in a way that they can be used with numpy arrays. or each coordinate
conversion its Jacobian is also implemented.
This functions are required to convert from the observation space (the coordinates in which the features are observed)
to the storage space (the coordinates in which the features are stored in the map). For instance, if the features are
observed in polar coordinates, but stored in cartesian coordinates, the conversion from polar to cartesian is required.

Polar To Cartesian
------------------
.. autofunction:: conversions.p2c

.. autofunction:: conversions.J_p2c

Cartesian To Polar
------------------

.. autofunction:: conversions.c2p

.. autofunction:: conversions.J_c2p

Spherical To Cartesian
----------------------

.. autofunction:: conversions.s2c

.. autofunction:: conversions.J_s2c

Cartesian To Spherical
----------------------

.. autofunction:: conversions.c2s

.. autofunction:: conversions.J_c2s

Identity Conversion
-------------------
.. autofunction:: conversions.v2v

.. autofunction:: conversions.J_v2v