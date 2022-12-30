## Introduction
<p align="justify">
This repository contains both the code and the report for Politectino di Milano's Passion in Action course *GPU 101*.
</p>

### Where can I find the material
The report and the CUDA code can be found in the spmv folder. For reasons of space, the test matrix can be found and downloaded at 
following link
https://tinyurl.com/gpumatrix

## How to get this repository
You can either download the code or type # in a terminal supporting git.

## Compilation

If you have already set up your machine for running CUDA code (so you have already installed NVIDIA's developers toolkit or you have access to an NVIDIA GPU via an online service), you'll just need to run 
```
make
```
in your Linux terminal: that script will automatically call the nvcc compiler with the default path to the code. If you want to make any changes regarding file paths or names please modify the script. 

## Running the code

To run the code simply type the following command and wait for the execution.
```
./spmv/spmv-gpu path_to_matrix num_of_threads
```
The execution first steps are pretty slow, as the matrix must be firstly read and stored in memory, so be aware that executing the code may take some time.


