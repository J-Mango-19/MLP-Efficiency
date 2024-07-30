# MLP-Efficiency
Evaluates the impact of various methods for improving the speed (memory access, training, inference) of neural nets using implementations in Numpy, OpenBLAS, and two of 
my own C programs, all benchmarked on the MNIST dataset. 

![all](assets/all.png)

Modern deep learning models frequently exceed 100 billion parameters and consume substantial time and resources during training, making efficient implementations vital.
CPU-based MLP implementations trained on the MNIST dataset offer a small scale version of this problem from which to evaluate the effectiveness of several methods to accelerate
deep learning tasks. 

## Usage

Quick start: run `python3 script.py` to watch all the models train in real time. Add the `-visualize` flag to recreate the charts seen in this file.

Run `python3 main.py` or `make` then `./mnist_nn` after changing to a program's directory to experiment with individual implementations.
Several flags controlling batch size, additional visualizations, training iterations and more are already included, and are viewable by running the program with `-h` after it.

## Takeaways

I began this project by aiming to improve the training speed of the NumPy implementation, assuming that the extra effort of rewriting the program in C would reward me with a 
faster program since it bypasses the Python interpreter with a directly compiled C executable. 

Of course, NumPy is a highly optimized library and the C program I wrote took far longer to train than the original NumPy. 
Deep learning is very matrix multiplication intensive, and profiling my program confirmed this: matrix multiplication alone accounted for 93% of the program runtime. 
After researching and experimenting with a number of optimization techniques for matrix multiplication, I wrote an optimized variant of the C program, which roughly matches
NumPy and will even outperform it, depending on which machine they're run on. 
Finally, I used OpenBLAS in a third C program to determine the theoretical best performance on this task.

## Optimization methods

Training on MNIST with minibatch (default 100) stochastic gradient descent is too small of a problem to reasonably simulate larger problems, so
the programs' default settings require a forward pass on all 60,000 patterns every 100 training steps to calculate train and test accuracy. These input matrices, being 6,000 
times larger than training batches, account for the bulk of computation time and optimizations are therefore directed at them.

The base C implementation holds data in 2d arrays and performs matrix multiplication using using the triple for loop method. The innermost loop iterates over the longest
(batch) dimension, contributing to very strong cache locality. 

The optimized C implementation further improves cache locality by storing matrices in 1D arrays. Further improvements proved challenging, since conventional efficient matrix
multiplication techniques (Strassen's algorithm, matrix blocking, matrix transposition) failed to outperform the original `for` loops on the unevenly shaped input matrix. 
Implementing SIMD instructions and efficient pointer arithmetic dramatically improved performance, but only when compiled without optimization flags. 
Inspection of the base and optimized matrix multiplication function machine code confirmed that SIMD instructions, efficient pointer arithmetic, and loop unrolling are 
automatically enabled when compiling with the -O3 flag, further narrowing opportunity for improvement.

What did significantly improve performance was introducing a multithreading option for large input matrices into the matrix multiplication function. Splitting the batch 
dimension calculations across 10 threads barely affected cache misses and parallelized the function, making it competitive with NumPy, which also utilizes multithreading.
Such dramatic performance gains of 10 threads on a handful of CPU cores is a strong predictor

Finally, to estimate the theoretical best performance on this problem, I benchmarked a version of the program using OpenBLAS for matrix multiplications, which unsurprisingly
outperformed other implementations. 


## Analysis

The following figures were computing using `perf stat -e cache-misses,cache-references,instructions,cycles ./mnist_nn -iterations 100 -nodisplay`

Cache misses (base): 21495784
21495784      cache-misses                     #    8.01% of all cache refs         
268371507      cache-references                                                      

Cache misses (optimized)
18940843      cache-misses                     #    7.22% of all cache refs         
262199995      cache-references                                             

Cache misses (CBLAS):
19435207      cache-misses                     #   15.00% of all cache refs         
129537933      cache-references                   

The base C implementation has the most of both cache references and misses. The optimized version likely reduces both of these for two reasons: First, the optimized implementation
stores matrices in 1-dimensional arrays rather than 2-dimensional, which means that even very large matrices are stored continguously in memory. Second, The optimized implementation
employs multithreading, which should utilize multiple CPU cores, each of which contributes an L1 and often an L2 cache, resulting in more total cache memory. 
CBLAS falls in between the others in cache misses, but has far less total references, likely due to the use of a technique such as Strassen's that reduces the number of 
memory accesses and other powerful algorithmic optimizations. Unsuprisingly, it's a balance between efficient algorithms and efficient memory access patterns that yields optimal
performance.

On that note, efficiently training deep learning models is dependent on more than just hardware oriented optimizations. A basic example of this is the usage of He weight
initialization rather than sampling from a uniform distribution. The neural networks in this project trained to 90% accuracy with He initilization in just 100-200 training steps, 
roughly 80% faster than the 500-1000 steps needed to reach 90% accuracy with weights sampled from a uniform distribution.

## Acknowledgements

I found the following resources invaluable during the course of this project:

[Python/NumPy MLP](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) This program forms the backbone of the NumPy implementation in this repo.

[Matrix Multiplication Optimization Lecture](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/resources/lecture-1-intro-and-matrix-multiplicati
on/) This lecture and pieces of other lectures from the same course established my understanding of optimizing matrix multiplication and introduced me to the most effective methods.

[BLISlab: A Sandbox for Optimizing GEMM](https://github.com/flame/blislab) Helped me learn through experimentation what works and what doesn't in optimizing matrix multiplication.





