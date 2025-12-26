# Development Guidelines

## Runtime Type Checking 🔍

mesa-frames includes optional runtime type checking using [beartype](https://github.com/beartype/beartype) for development and debugging purposes. This feature helps catch type-related errors early during development and testing.

!!! tip "Automatically Enabled"
    Runtime type checking is **automatically enabled** in the following scenarios:

    - **Hatch development environment** (`hatch shell dev`) — via `pyproject.toml` configuration
    - **VS Code debugging** — when using the debugger (`F5` or "Python Debugger: Current File")
    - **VS Code testing** — when running tests through VS Code's testing interface

    No manual setup required in these environments!

### Development Environment Setup

#### Option 1: Hatch Development Environment (Recommended)

The easiest way to enable runtime type checking is to use Hatch's development environment:

```bash
# Enter the development environment (auto-enables runtime type checking)
hatch shell dev

# Verify it's enabled
python -c "import os; print('Runtime type checking:', os.getenv('MESA_FRAMES_RUNTIME_TYPECHECKING'))"
# → Runtime type checking: true
```

#### Option 2: Manual Environment Variable

For other development setups, you can manually enable runtime type checking:

Runtime type checking can be enabled by setting the `MESA_FRAMES_RUNTIME_TYPECHECKING` environment variable:

```bash
export MESA_FRAMES_RUNTIME_TYPECHECKING=1
# or
export MESA_FRAMES_RUNTIME_TYPECHECKING=true
# or
export MESA_FRAMES_RUNTIME_TYPECHECKING=yes
```

### Usage Examples

!!! info "Automatic Activation"
    If you're using **Hatch dev environment**, **VS Code debugging**, or **VS Code testing**, runtime type checking is already enabled automatically. The examples below are for manual activation in other scenarios.

#### For Development and Testing

```bash
# Enable runtime type checking for testing
MESA_FRAMES_RUNTIME_TYPECHECKING=1 uv run pytest

# Enable runtime type checking for running scripts
MESA_FRAMES_RUNTIME_TYPECHECKING=1 uv run python your_script.py
```

#### In Your IDE or Development Environment

**VS Code** (Already Configured):

- **Debugging**: Runtime type checking is automatically enabled when using VS Code's debugger
- **Testing**: Automatically enabled when running tests through VS Code's testing interface
- **Manual override**: You can also add it manually in `.vscode/settings.json`:

    ```json
    {
        "python.env": {
            "MESA_FRAMES_RUNTIME_TYPECHECKING": "1"
        }
    }
    ```

**PyCharm**:
In your run configuration, add the environment variable:

```bash
MESA_FRAMES_RUNTIME_TYPECHECKING=1
```

### How It Works

When enabled, the runtime type checking system:

1. **Automatically instruments** all mesa-frames packages with beartype decorators
2. **Validates function arguments** and return values at runtime
3. **Provides detailed error messages** when type mismatches occur
4. **Helps catch type-related bugs** during development

### Requirements

Runtime type checking requires the optional `beartype` dependency:

```bash
# Install beartype for runtime type checking
uv add beartype
# or
pip install beartype
```

!!! note "Optional Dependency"
    If `beartype` is not installed and runtime type checking is enabled, mesa-frames will issue a warning and continue without type checking.

### Performance Considerations

### Numba Cache

Some fast paths use Numba JIT compilation. By default, mesa-frames enables Numba's
on-disk cache to speed up subsequent runs by reusing compiled artifacts.

- Enable (default): `MESA_FRAMES_NUMBA_CACHE=1`
- Disable: `MESA_FRAMES_NUMBA_CACHE=0`

When enabled, Numba writes cache files into the module's `__pycache__` directory.
On read-only installs this may warn and/or fall back to no cache, which is
usually acceptable. For CI, tests, and benchmarks, it's recommended to disable
the cache to avoid unexpected writes.

!!! warning "Development Only"
    Runtime type checking adds significant overhead and should **only be used during development and testing**. Do not enable it in production environments.

The overhead includes:

- Function call interception and validation
- Type checking computations at runtime
- Memory usage for type checking infrastructure

### When to Use Runtime Type Checking

✅ **Automatically enabled (recommended):**

- Hatch development environment (`hatch shell dev`)
- VS Code debugging sessions
- VS Code test execution
- Contributing to mesa-frames development

✅ **Manual activation (when needed):**

- Development and debugging in other IDEs
- Writing new features outside VS Code
- Running unit tests from command line
- Troubleshooting type-related issues

❌ **Not recommended for:**

- Production deployments
- Performance benchmarking
- Large-scale simulations
- Final model runs

### Troubleshooting

If you encounter issues with runtime type checking:

1. **Check beartype installation:**

   ```bash
   uv run python -c "import beartype; print(beartype.__version__)"
   ```

2. **Verify environment variable:**

   ```bash
   echo $MESA_FRAMES_RUNTIME_TYPECHECKING
   ```

3. **For automatic configurations:**
   - **Hatch dev**: Ensure you're in the dev environment (`hatch shell dev`)
   - **VS Code debugging**: Check that the debugger configuration in `.vscode/launch.json` includes the environment variable
   - **VS Code testing**: Verify that `.env.test` file exists and contains `MESA_FRAMES_RUNTIME_TYPECHECKING=true`

4. **Check for warnings** in your application logs

5. **Disable temporarily** if needed:

   ```bash
   unset MESA_FRAMES_RUNTIME_TYPECHECKING
   ```

!!! tip "Pro Tip"
    Runtime type checking is particularly useful when developing custom AgentSet implementations or working with complex DataFrame operations where type safety is crucial.
