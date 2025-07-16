import os

# Enable runtime type checking if requested via environment variable
if os.getenv("MESA_FRAMES_RUNTIME_TYPECHECKING", "").lower() in ("1", "true", "yes"):
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except ImportError:
        import warnings

        warnings.warn(
            "MESA_FRAMES_RUNTIME_TYPECHECKING is enabled but beartype is not installed.",
            ImportWarning,
            stacklevel=2,
        )
