Space
=====
This page provides a high-level overview of possible space objects for mesa-frames models.

.. currentmodule:: mesa_frames

Quick intro
-----------



Currently we only support the ``Grid``. Typical usage:

- Construct ``Grid(model, (width, height))`` and use ``place``/ ``move`` helpers to update agent positional columns.
- Use neighbourhood queries to produce masks or index lists and then apply vectorised updates to selected rows.

Minimal example
---------------

.. code-block:: python

    from mesa_frames import Model, Grid, AgentSet
    import polars as pl

    class P(AgentSet):
        pass

    class M(Model):
        def __init__(self):
            super().__init__()
            self.space = Grid(self, (10, 10))
            self.sets += P(self)
            self.space.place_to_empty(self.sets)

    m = M()
    m.space.move_to_available(m.sets)


API reference
-------------

.. autoclass:: Grid
    :members:
    :inherited-members:
    :autosummary:
    :autosummary-nosignatures: