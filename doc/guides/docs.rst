==========================
Working with Documentation
==========================

We write Vivarium's documentation in plain text that utilizes the
`reStructured Text <https://www.sphinx-doc.org/rest.html>`_ markup
language. You can compile it to HTML with `Sphinx
<https://www.sphinx-doc.org>`_, and you can also read it as plain text.

---------------------
Reading Documentation
---------------------

You're welcome to read the plain text documentation in this folder, but
you'll probably enjoy the pretty HTML version more.
`Read the Docs <https://readthedocs.org>`_ hosts the compiled HTML here:
https://wc-vivarium.rtfd.io/

If you want to generate the HTML documentation yourself, check out the
instructions on building documentation :ref:`below <building-docs>`.

---------------------
Writing Documentation
---------------------

Where to Write
==============

We write four kinds of documentation for Vivarium:

* Tutorials: These are step-by-step instructions that walk the reader
  through doing something with Vivarium. We store these in
  ``doc/tutorials``. Make sure to list all tutorials in
  ``doc/tutorials/index.rst`` so that they appear in the sidebar and the
  list of tutorials.
* Guides: These dive into the details of Vivarium and should
  be comprehensive. We store guides in ``doc/guides`` and list them in
  ``doc/guides/index.rst``. Guides should focus on the conceptual
  aspects of Vivarium, leaving technical details to the API reference.
* References: Reference material should cater to users who already know
  what they're looking for and just need to find it. For example, a user
  looking up a particular process or term. Our reference material
  consists of a glossary and an API reference. The glossary is stored in
  ``doc/glossary.rst``, while the API reference is auto-generated from
  docstrings in the code. These docstrings can take advantage of all the
  reStructuredText syntax we use elsewhere in Vivarium. Eventually, we
  will remove from the reference material the stubs for functions that
  aren't user-facing and the auto-generated titles on each page.

    * For an example of reference documentation that defines an API, see
      :py:mod:`vivarium.processes.death`. For an example of
      documentation that explains how to use a process, look at
      :py:mod:`vivarium.processes.metabolism`.

      .. note::
          From the compiled HTML reference documentation, you can click
          on ``[source]`` to see the source code, including the
          docstrings. This can be helpful for looking up
          reStructuredText syntax.

      .. WARNING:: For each class, include at most one of the class and
          constructor docstrings. They are concatenated when the HTML is
          compiled, so you can provide either one.

          .. code-block:: python

                class MyClass:
                    '''This is the class docstring'''

                    def __init__(self):
                        '''This is the constructor docstring'''

Glossary vs Guide vs API Reference
----------------------------------

In the guide, describe the concept and perhaps our rationale behind any
design choices we made. Link terms to the glossary, which succinctly
describes the term and links to relevant API reference pages and guides.
In the API reference, describe the technical details.

We try to keep technical details in the API reference because the API
reference is built from docstrings. Since these docstrings live
alongside the code, they are more likely to be kept up-to-date than a
separate guide.

Pointers for Technical Writing
==============================

Here are resources for writing good documentation and technical writing
in general:

* http://jacobian.org/writing/what-to-write/
* https://www.writethedocs.org/

.. todo:: Flesh out these pointers based on what I learn while writing

Style Guide
===========

Here we document the stylistic decisions we have made for this
documentation:

* We use first-person plural pronouns to refer to ourselves (e.g. "We
  decided").
* We write tutorials in the second-person, future tense, for example
  "First, you'll need to install". We also frequently use the imperative
  ("Install this").
* We use the following admonitions. We don't want to overload our users
  with admonitions, so we don't use any others.

    * We warn users about potential problems with warning admonitions.
      These often describe important steps that we think users might forget.

      .. WARNING::

         ``.. WARNING::``

    * We use notes to highlight important points. These should *not* be
      used for asides that aren't important enough to integrate directly
      into the text.

      .. note::

         ``.. note::``

    * We give users helpful tips using the tip admonition. These help
      highlight tips that some users might not use but that will help
      users who are debugging problems.

      .. tip::

         ``.. tip::``

    * We use danger admonitions for the most critical warnings. Use
      these sparingly.

      .. DANGER::

         ``.. DANGER::``

* We use `Vale <https://errata-ai.gitbook.io/vale/>`_ to lint our
  documentation. You can run the linter by executing ``doc/test.sh``.
  This linter checks some Vivarium-specific naming and capitalization
  conventions. It also runs the ``proselint`` and ``write-good``
  linters, which check for generally good style.


.. _building-docs:

Building the Documentation
==========================

To build the documentation, we will use Sphinx to generate HTML files
from plain text. Here are stepwise instructions:

#. (optional) Create a virtual environment for the
   documentation-building packages. You might want this to be separate
   from the environment you use for the rest of Vivarium.
#. Install dependencies:

   .. code-block:: console

        $ pip install -r doc/requirements.txt

#. Build the HTML!

   .. code-block:: console

        $ cd doc
        $ make html

   Your HTML will now be in ``doc/_build/html``. To view it, open
   ``doc/_build/html/index.html`` in a web browser.

.. todo:: Add instructions for working with readthedocs.io
