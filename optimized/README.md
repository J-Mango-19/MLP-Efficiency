efficiency boosts:
0: 2d matrix program w no boosts
1: flattened all matrices into 1d
1.5: Utilized pointers in matrix multiplication
2: Loop unrolling in matrix multiplication
3: 

With optimizations, this actually makes the program slower than the simple 2d array implementation

* Assume all runs are 500 iterations and 100 batch size

0:   -O0: 33.2 training and 4.25 full forward
     -O1: 9.2 training and 1.17 full forward

1:   -O0: 34.8 training and 4.7 full forward
     -O1: 21.4 training and 3.0 full forward

1.5: -O0: 18.7 training and 2.4 full forward
     -O1: 5.14 training and 0.65 full forward

2:   -O0: 15.9 training and 2.0 full forward
     -O1: 4.67 training and 0.60 full forward
