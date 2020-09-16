# Usage: test.sh type
#   type: Either "ci" in which case only errors are raised, or "dev"
#       (default) in which case warnings are shown as well

type=${1:-"dev"}

if [[ $type == "dev" ]]; then
    alertLevel="warning"
elif [[ $type == "ci" ]]; then
    alertLevel="error"
else
    echo "ERROR: type '$type' not recognized"
    exit 1
fi

# Lint
vale --minAlertLevel "$alertLevel" --glob "*.rst" _static getting_started.rst \
    guides index.rst reference/glossary.rst reference/index.rst tutorials
