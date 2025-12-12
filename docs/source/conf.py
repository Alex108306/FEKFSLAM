# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('/Users/pere/PycharmProjects/PRPY'))

project = 'prpy: Probabilistic Robot Localization Python Library'
copyright = '2023, Pere Ridao'
author = 'Pere Ridao'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'sphinxcontrib.plantuml',
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'private-members': False,
    'autoclass_content': 'both',
    'inheritance_diagram': True,
}

autodoc_member_order = 'bysource'


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'bizstyle'

html_theme_options = {
    "sidebarwidth" :500,
}

html_static_path = ['_static']


