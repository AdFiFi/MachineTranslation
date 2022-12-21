#!/bin/bash

export PYTHONUNBUFFERED=1

# shellcheck disable=SC2164
cd ../
python  main.py --do_evaluate --do_parallel