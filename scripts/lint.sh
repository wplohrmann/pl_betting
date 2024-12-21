#!/bin/bash

set -e

black src

mypy src
