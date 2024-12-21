#!/bin/bash

set -e

black src

mypy src

if ! git diff --quiet; then
    echo "Linting caused changes in the files. Please review and commit the changes."
    exit 1
fi
