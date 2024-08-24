import os

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information -----------------------------------------------------
project = "mesa-frames"
author = "Adam Amer"
copyright = f"2023, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "autodocsumm",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False

# -- Extension settings ------------------------------------------------------
# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://pola-rs.github.io/polars/py-polars/html/", None),
}

# numpydoc settings
numpydoc_show_class_members = False

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Custom configurations ---------------------------------------------------
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {"special-members": True, "exclude-members": "__weakref__"}

# -- GitHub link and user guide settings -------------------------------------
github_root = "https://github.com/adamamer20/mesa-frames"
web_root = "https://adamamer20.github.io/mesa-frames"

html_theme_options = {
    "external_links": [
        {
            "name": "User guide",
            "url": f"{web_root}/user-guide/",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_root,
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
}
