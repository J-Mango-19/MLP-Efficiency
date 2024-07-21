Accuracy boosts:
0: He initialization: the network learns far faster, especially in the beginning of training & achieves higher final train & test accuracy. ~96%->~98% train and ~95%->~96% test 

efficiency boosts:
0: 2d matrix program w no boosts
1: flattened all matrices into 1d
1.5: Utilized pointers in matrix multiplication
2: Loop unrolling in matrix multiplication
3: Storing inner loop variables in registers
4: SIMD using avx
5: Improved pointer arithmetic and general program efficiency

With optimizations, this actually makes the program slower than the simple 2d array implementation

* Assume all runs are 500 iterations and 100 batch size

0:   -O0: 33.2 training and 4.25 full forward
     -O1: 9.2 training and 1.17 full forward
     -O3: 5.20 training and 0.66 full forward

1:   -O0: 34.8 training and 4.7 full forward
     -O1: 21.4 training and 3.0 full forward

1.5: -O0: 18.7 training and 2.4 full forward
     -O1: 5.14 training and 0.65 full forward

2:   -O0: 15.9 training and 2.0 full forward
     -O1: 4.67 training and 0.60 full forward

3:   -O0: 15.2 training and 1.95 full forward
     -O1: 4.0 training and 0.50 full forward

4:   -O0: 9.36 training and 1.12 full forward
     -O1: 2.31 training and 0.33 full forward
     -O2: 2.20 training and 0.32 full forward
     -O3: 2.23 training and 0.32 full forward

5:   -O0: 7.92 training and 0.98 full forward
     -O1: 2.39 training and 0.32 full forward

