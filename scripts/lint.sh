#!/bin/bash

set -e

black $(git ls-files | "grep" "\.py$")
