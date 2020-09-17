# Usage: lint.sh type
#   type: Either "ci" in which case only errors are raised, or "dev"
#       (default) in which case warnings are shown as well

set -e

type="$1"
dirpath=$(dirname "$0")
curdir=$(pwd)

if [[ $type == "dev" ]]; then
    alertLevel="warning"
elif [[ $type == "ci" ]]; then
    alertLevel="error"
else
    echo "ERROR: type '$type' not recognized"
    exit 1
fi

# Lint
cd "$dirpath"
vale --minAlertLevel "$alertLevel" --glob "*.rst" _static getting_started.rst \
    guides index.rst reference/glossary.rst reference/index.rst tutorials
cd "$curdir"
