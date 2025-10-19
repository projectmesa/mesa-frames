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

.. tab-set::

    .. tab-item:: Overview

        .. rubric:: Lifecycle / Core

        .. autosummary::
            :nosignatures:
            :toctree:

            Grid.__init__
            Grid.copy

        .. rubric:: Placement & Movement

        .. autosummary::
            :nosignatures:
            :toctree:

            Grid.place_agents
            Grid.move_agents
            Grid.place_to_empty
            Grid.place_to_available
            Grid.move_to_empty
            Grid.move_to_available

        .. rubric:: Sampling & Queries

        .. autosummary::
            :nosignatures:
            :toctree:

            Grid.get_neighbors
            Grid.get_directions
            Grid.get_distances
            Grid.sample_cells
            Grid.random_pos
            Grid.is_empty
            Grid.is_available
            Grid.is_full

        .. rubric:: Accessors & Metadata

        .. autosummary::
            :nosignatures:
            :toctree:

            Grid.dimensions
            Grid.neighborhood_type
            Grid.torus
            Grid.remaining_capacity
            Grid.agents
            Grid.model
            Grid.random

    .. tab-item:: Full API

        .. autoclass:: Grid
            :autosummary:
            :autosummary-nosignatures:
