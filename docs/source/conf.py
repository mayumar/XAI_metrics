# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XAI_metrics'
copyright = '2026, Maria Teresa Alba Rueda'
author = 'Maria Teresa Alba Rueda'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_mock_imports = ["quantus", "xplique"]
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']


autodoc_type_aliases = {
    "BaseMetric": "xai_metrics.base.base.BaseMetric",
    "MetricContext": "xai_metrics.base.base.MetricContext",
    "MetricSkipped": "xai_metrics.base.base.MetricSkipped",
}
