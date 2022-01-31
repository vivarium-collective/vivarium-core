============================
Versioning and API Stability
============================

We want to keep our API stable so that you can rely on it. To help
clearly communicate how our API is changing, we strive to following the
guidelines below when releasing new versions of Vivarium.

We follow `semantic versioning <https://semver.org/>`_, which in brief
means the following:

* Our version numbers have the form ``major.minor.patch`` (e.g.
  ``1.2.3``).
* We increment ``major`` for breaking changes to Vivarium's supported
  application programming interface (API). Upgrading to a new major
  release may break your code.
* We increment ``minor`` when we release new functionality. New minor
  releases should be backwards-compatible with the previous version of
  Vivarium.
* We increment ``patch`` for bug fixes that neither release new
  functionality nor introduce breaking changes.

Note that the above rules only apply to Vivarium's supported API. If you
use unsupported features, they may break with a new minor or patch
release. Our supported API consists of all public, documented interfaces
that are not marked as being experimental in their documentation. Note
that attributes beginning with an underscore (``_``) are private. The
following are not considered breaking API changes:

* Adding to a function a parameter with a default value (i.e. an
  optional parameter).
* Adding a new key to a dictionary. For example, expanding the
  ``Composite`` dictionary to include an extra key.
* When a function accepts a dictionary as an argument, adding more,
  optional keys to that dictionary. For example, letting the user
  specify a new key in a configuration dictionary.

Changes to the supported API will be reflected in Vivarium's versioning.
We will also try to mark interfaces as deprecated in their documentation
and raise warnings at least one release before actually removing them.
