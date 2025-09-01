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

