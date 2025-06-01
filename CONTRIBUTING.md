# Contributing to mesa-frames 🚀

Thank you for taking the time to contribute to **mesa-frames**! Since the project is still in its early stages, we warmly welcome contributions that will help shape its development. 🎉

For a more general and comprehensive guide, please refer to [mesa's main contribution guidelines](https://github.com/projectmesa/mesa/blob/main/CONTRIBUTING.md). 📜

## Project Roadmap 🗺️

Before contributing, we recommend reviewing our [roadmap](https://projectmesa.github.io/mesa-frames/roadmap/) file to understand the project's current priorities, upcoming features, and long-term vision. This will help ensure your contributions align with the project's direction.

## How to Contribute 💡

### 1. Prerequisite Installations ⚙️

Before you begin contributing, ensure that you have the necessary tools installed:

- **Install Python** (at least the version specified in `requires-python` of `pyproject.toml`). 🐍
- We recommend using a virtual environment manager like:
  - [Astral's UV](https://docs.astral.sh/uv/#installation) 🌟
  - [Hatch](https://hatch.pypa.io/latest/install/) 🏗️
- Install **pre-commit** to enforce code quality standards before pushing changes:
  - [Pre-commit installation guide](https://pre-commit.com/#install) ✅
  - [More about pre-commit hooks](https://stackoverflow.com/collectives/articles/71270196/how-to-use-pre-commit-to-automatically-correct-commits-and-merge-requests-with-g)
- If using **VS Code**, consider installing these extensions to automatically enforce formatting:
  - [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) – Python linting & formatting 🐾
  - [Markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) – Markdown linting (for documentation) ✍️
  - [Git Hooks](https://marketplace.visualstudio.com/items?itemName=lakshmikanthayyadevara.githooks) – Automatically runs & visualizes pre-commit hooks 🔗

---

### 2. Contribution Process 🛠️

#### **Step 1: Choose an Issue** 📌

- Pick an existing issue or create a new one if necessary.
- Ensure that your contribution aligns with the project's goals.

#### **Step 2: Set Up Your Local Repository** 💻

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine:

   ```sh
   git clone https://github.com/YOUR_USERNAME/mesa-frames.git
   ```

3. **Create a new branch** with a descriptive name:

   ```sh
   git checkout -b feature-name
   ```

4. **Prevent merge commit clutter** by setting rebase mode:

   ```sh
   git config pull.rebase true
   ```

#### **Step 3: Install Dependencies** 📦

It is recommended to set up a virtual environment before installing dependencies.

- **Using UV**:

  ```sh
  uv add --dev .[dev]
  ```

- **Using Hatch**:

  ```sh
  hatch env create dev
  ```

- **Using Standard Python**:

  ```sh
  python3 -m venv myenv
  source myenv/bin/activate  # macOS/Linux
  myenv\Scripts\activate    # Windows
  pip install -e ".[dev]"
  ```

#### **Step 4: Make and Commit Changes** ✨

1. Make necessary edits and save the code.
2. **Add and commit** your changes with meaningful commit messages:

   ```sh
   git add FILE_NAME
   git commit -m "Fix issue X: Brief description of the fix"
   ```

   - Keep commits **small and focused** on a single logical change.
   - Follow [Tim Pope’s commit message guidelines](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html). 📝

#### **Step 5: Code Quality and Testing** ✅

- **Run pre-commit hooks** to enforce code quality standards:

  ```sh
  pre-commit run
  ```

- **Run tests** to ensure your contribution does not break functionality:

  ```sh
  pytest --cov
  ```

  - If using UV: `uv run pytest --cov`

- **Optional: Enable runtime type checking** during development for enhanced type safety:

  ```sh
  MESA_FRAMES_RUNTIME_TYPECHECKING=1 uv run pytest --cov
  ```

  !!! tip "Automatically Enabled"
      Runtime type checking is automatically enabled in these scenarios:

      - **Hatch development environment** (`hatch shell dev`)
      - **VS Code debugging** (when using the debugger)
      - **VS Code testing** (when running tests through VS Code's testing interface)

      No manual setup needed in these environments!

  For more details on runtime type checking, see the [Development Guidelines](https://projectmesa.github.io/mesa-frames/development/).

#### **Step 6: Documentation Updates (If Needed)** 📖

- If you add a new feature, update the documentation accordingly.
- We use **[MKDocs](https://www.mkdocs.org/)** for documentation:
  - Modify or create markdown files in the `docs/` folder.
  - Preview your changes by running:

    ```sh
    mkdocs serve
    uv run mkdocs serve #If using uv
    ```

  - Open `http://127.0.0.1:8000` in your browser to verify documentation updates.

#### **Step 7: Push Changes and Open a Pull Request (PR)** 🚀

1. **Push your changes** to your fork:

   ```sh
   git push origin feature-name
   ```

2. **Open a pull request (PR)**:
   - Follow [GitHub’s PR guide](https://help.github.com/articles/creating-a-pull-request/).
   - Link the issue you are solving in the PR description.

---

Thank you again for your contribution! 🎉
