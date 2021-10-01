# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import datetime
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath('..'))

from docutils.nodes import Text
from sphinx.addnodes import pending_xref
from sphinx.ext import apidoc
from sphinx.ext.intersphinx import missing_reference

from setup import VERSION


# -- Project information -----------------------------------------------------

project = 'Vivarium Core'
copyright = '2018-{}, The Vivarium Core Authors'.format(
    datetime.now().year)
author = 'The Vivarium Core Authors'

# The full version, including alpha/beta/rc tags
release = 'v{}'.format(VERSION)
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'venv']

# Causes warnings to be thrown for all unresolvable references. This
# will help avoid broken links.
nitpicky = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for extensions --------------------------------------------------

# -- sphinx.ext.intersphinx options --
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/3.3.1/', None),
    'networkx': ('http://networkx.org/documentation/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'shapely': ('https://shapely.readthedocs.io/en/latest/', None),
    'pint': ('https://pint.readthedocs.io/en/stable/', None),
}

# -- sphinx.ext.napoleon options --
# Map from the alias to a tuple of the actual ref and the text to
# display.
reftarget_aliases = {
    type_name: ('vivarium.core.types.{}'.format(type_name), type_name)
    for type_name in (
        'HierarchyPath', 'Topology', 'Schema', 'State', 'Update',
        'CompositeDict')
}
reftarget_aliases.update({
    type_name: ('typing.{}'.format(type_name), type_name)
    for type_name in (
        'Any', 'Dict', 'Tuple', 'Union', 'Optional', 'Callable', 'List')
})


# -- sphinx.ext.autodoc options --
autodoc_inherit_docstrings = False
# The Python dependencies aren't really required for building the docs
autodoc_mock_imports = [
    'arpeggio', 'cobra', 'matplotlib',
    'mpl_toolkits', 'networkx', 'parsimonious', 'pygame', 'pymongo',
    'arrow',
]
# Concatenate class and __init__ docstrings
autoclass_content = 'both'

def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if name.startswith('test_'):
        return True
    return None


cur_dir = os.path.abspath(os.path.dirname(__file__))
notebooks_dst = os.path.join(cur_dir, 'notebooks')
notebooks_src = os.path.join(cur_dir, '..', 'notebooks')
if os.path.exists(notebooks_dst):
    shutil.rmtree(notebooks_dst)
shutil.copytree(notebooks_src, notebooks_dst)


# -- Custom Extensions -------------------------------------------------


# This function is adapted from a StackOverflow answer by Oleg Höfling
# at https://stackoverflow.com/a/62301461. Per StackOverflow's licensing
# terms, it is available under a CC-BY-SA 4.0 license
# (https://creativecommons.org/licenses/by-sa/4.0/).
def resolve_internal_aliases(_, doctree):
    pending_xrefs = doctree.traverse(condition=pending_xref)
    for node in pending_xrefs:
        alias = node.get('reftarget')
        if alias is not None and alias in reftarget_aliases:
            resolved_ref, text = reftarget_aliases[alias]
            node['reftarget'] = resolved_ref
            text_node = next(iter(
                node.traverse(lambda n: n.tagname == '#text')))
            text_node.parent.replace(text_node, Text(text, ''))


# This function is adapted from a StackOverflow answer by Oleg Höfling
# at https://stackoverflow.com/a/62301461. Per StackOverflow's licensing
# terms, it is available under a CC-BY-SA 4.0 license
# (https://creativecommons.org/licenses/by-sa/4.0/).
def resolve_intersphinx_aliases(app, env, node, contnode):
    alias = node.get('reftarget')
    if alias is not None and alias in reftarget_aliases:
        resolved_ref, text = reftarget_aliases[alias]
        node['reftarget'] = resolved_ref
        text_node = next(iter(
            contnode.traverse(lambda n: n.tagname == '#text')))
        text_node.parent.replace(text, Text(text, ''))
        return missing_reference(app, env, node, contnode)
    return None


# This function is adapted from a StackOverflow answer by Oleg Höfling
# at https://stackoverflow.com/a/62301461. Per StackOverflow's licensing
# terms, it is available under a CC-BY-SA 4.0 license
# (https://creativecommons.org/licenses/by-sa/4.0/).
def setup(app):
    app.connect('doctree-read', resolve_internal_aliases)
    app.connect('missing-reference', resolve_intersphinx_aliases)
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)