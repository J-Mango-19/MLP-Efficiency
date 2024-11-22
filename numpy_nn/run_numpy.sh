#!/bin/bash

# try running main.py with different versions of python, hopefully one of these works on most machines

python main.py $@ 2>/dev/null

if [ "$?" -eq "0" ]; then
    exit 0
fi

python3 main.py $@ 2>/dev/null

if [ "$?" -eq "0" ]; then
    exit 0
fi

python3.11 main.py $@ 2>/dev/null

if [ "$?" -eq "0" ]; then
    exit 0
fi

python3.12 main.py $@ 2>/dev/null

if [ "$?" -ne "0" ]; then
    echo "Could not find python, python3, python3.11, or python3.12 to run numpy_nn"
    exit 1
fi


