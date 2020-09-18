# Create and publish a new version by creating and pushing a git tag for
# the version and publishing the version to PyPI. Also perform some
# basic checks to avoid mistakes in releases, for example tags not
# matching PyPI.
# Usage: ./release.sh 0.0.1

set -e

version="$1"

# Check version is valid
setup_py_version="$(python setup.py --version)"
if [ "$setup_py_version" != "$version" ]; then
    echo "setup.py has version $setup_py_version, not $version."
    echo "Aborting."
    exit 1
fi

# Check working directory is clean
if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting."
    exit 1
fi

# Check that we are on master
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "master" ]; then
    echo "You are on $branch but should be on master for releases."
    echo "Aborting."
    exit 1
fi

# Create and push git tag
git tag -m "Version v$version" "v$version"
git push --tags

# Create and publish package
rm -rf dist
python setup.py sdist
twine upload dist/*

echo "Version v$version has been published on PyPI and has a git tag."
