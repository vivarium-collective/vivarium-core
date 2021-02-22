#!/bin/sh
# Run all checks on this project.

set -eu

echo === mypy ===
runscripts/mypy.sh

echo === pylint ===
runscripts/pylint.sh

echo === pytest ===
runscripts/pytest.sh
