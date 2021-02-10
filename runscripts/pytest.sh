#!/bin/sh
# Run pytest tests on this project.
# Configured by `pytest.ini` and `.coveragerc`.

set -eu

pytest --cov=vivarium
