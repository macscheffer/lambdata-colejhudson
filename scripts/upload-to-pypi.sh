#!/usr/bin/env bash

PACKAGE=$(ls dist/ | grep tar | tail -n 1 | awk '{ print $9 }')
twine upload -r pypitest dist/${PACKAGE}
