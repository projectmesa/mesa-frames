"""Utility functions for mesa_frames."""


def copydoc(fromfunc, sep="\n"):
    """Copy the docstring of function or class.

    https://stackoverflow.com/a/13743316
    """

    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ == None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func

    return _decorator


def camel_case_to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case.

    Parameters
    ----------
    name : str
        The camelCase string to convert.

    Returns
    -------
    str
        The converted snake_case string.

    Examples
    --------
    >>> camel_case_to_snake_case("ExampleAgentSetPolars")
    'example_agent_set_polars'
    >>> camel_case_to_snake_case("getAgentData")
    'get_agent_data'
    """
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
