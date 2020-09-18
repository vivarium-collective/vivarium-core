# Usage: test.sh type
#   type: Either "ci" in which case only errors are raised, or "dev"
#       (default) in which case warnings are shown as well

set -e

type="$1"
dirpath=$(dirname "$0")
curdir=$(pwd)

sphinxopts=""

if [[ $type == "dev" ]]; then
    # Do nothing
    :
elif [[ $type == "ci" ]]; then
    sphinxopts="$sphinxopts -W"
else
    echo "ERROR: type '$type' not recognized"
    exit 1
fi

cd "$dirpath"
make html SPHINXOPTS="$sphinxopts"
cd "$curdir"
