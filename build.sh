#!/usr/bin/bash
echo "COMPILING"
gcc ./spmv/spmv-csr.c -O3 -Wunused-result -o spmv_run
echo "DONE"
