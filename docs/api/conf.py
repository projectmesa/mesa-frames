import os

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information -----------------------------------------------------
project = "mesa-frames"
author = "Project Mesa, Adam Amer"
copyright = f"{datetime.now().year}, {author}"

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
# Hide objects (classes/methods) from the page Table of Contents
toc_object_entries = False  # NEW: stop adding class/method entries to the TOC


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_logo = (
    "https://raw.githubusercontent.com/projectmesa/mesa/main/docs/images/mesa_logo.png"
)
html_favicon = (
    "https://raw.githubusercontent.com/projectmesa/mesa/main/docs/images/mesa_logo.ico"
)

# Add custom branding CSS/JS (mesa_brand) to static files
html_css_files = [
    # Shared brand variables then theme adapter for pydata
    "brand-core.css",
    "brand-pydata.css",
]

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
add_module_names = False
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__weakref__,__dict__,__module__,__annotations__,__firstlineno__,__static_attributes__,__abstractmethods__,__slots__",
}


# -- GitHub link and user guide settings -------------------------------------
github_root = "https://github.com/projectmesa/mesa-frames"
web_root = "https://projectmesa.github.io/mesa-frames"

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
        {
            "name": "Matrix",
            "url": "https://matrix.to/#/#project-mesa:matrix.org",
            "icon": "fa-solid fa-comments",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
