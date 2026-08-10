"""Sphinx configuration for the UniTSFM documentation."""

from __future__ import annotations

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "UniTSFM"
copyright = f"{date.today().year}, Sourav Roy"
author = "Sourav Roy"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_rtype = True

# Autosummary
autosummary_generate = True

# Todo
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = f"UniTSFM v{release}"
html_logo = None
html_favicon = None
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "figure_align": "htbp",
}
latex_documents = [
    (
        "index",
        "uniftsm.tex",
        "UniTSFM Documentation",
        "Sourav Roy",
        "manual",
    ),
]

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# -- Exclude patterns --------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Suppress warnings -------------------------------------------------------

suppress_warnings = ["autodoc.import_object"]
