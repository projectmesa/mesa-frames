"""
mesa-frames abstract components.

This package contains abstract base classes and mixins that define the core
interfaces and shared functionality for the mesa-frames extension.

Classes:
    agents.py:
        - AgentContainer: Abstract base class for agent containers.
        - AgentSetDF: Abstract base class for agent sets using DataFrames.

    mixin.py:
        - CopyMixin: Mixin class providing fast copy functionality.
        - DataFrameMixin: Mixin class defining the interface for DataFrame operations.

    space.py:
        - SpaceDF: Abstract base class for all space classes.
        - DiscreteSpaceDF: Abstract base class for discrete space classes (Grids and Networks).
        - GridDF: Abstract base class for grid classes.

These abstract classes and mixins provide the foundation for the concrete
implementations in mesa-frames, ensuring consistent interfaces and shared
functionality across different backend implementations (currently support only Polars).

Usage:
    These classes are not meant to be instantiated directly. Instead, they
    should be inherited by concrete implementations in the mesa-frames package.

    For example:

    from mesa_frames.abstract import AgentSetDF, DataFrameMixin

    class ConcreteAgentSet(AgentSetDF):
        # Implement abstract methods here
        ...

Note:
    The abstract classes use Python's ABC (Abstract Base Class) module to define
    abstract methods that must be implemented by concrete subclasses.

For more detailed information on each class, refer to their individual docstrings.
"""
