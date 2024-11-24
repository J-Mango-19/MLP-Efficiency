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
    echo "Failed to run numpy_nn."
    echo "Possible cause is that run_numpy.sh couldn't find a python version. Tried python, python3, python3.11, python3.12."
    exit 1
fi


