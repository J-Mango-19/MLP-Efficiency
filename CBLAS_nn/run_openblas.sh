#!/bin/bash

export OPENBLAS_NUM_THREADS=8

if [ ! -e "mnist_nn" ]; then
    echo "executable not found. making... "
    make clean
    make
fi

if [ "$?" -ne "0" ]; then
    echo ""
    echo "Compilation of Openblas C implementation failed"
    exit 1
fi

LD_PRELOAD=~/local/openblas/lib/libopenblas.so ./mnist_nn $@
