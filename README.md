"Optimizing a Neural Network for Efficiency"

Next steps:
DONE 0) look into replacing small matrix multiplication function with the one in matmul/clean folder. Test both of them there first
DONE 1) put he initialization in base folder
DONE 2) replace timing functions in base folder
DONE 3) remove 'optimized' folder (probably rename 'improved' to 'optimized' after this)
DONE 4) move python folder into project
DONE 5) put he initialization in python folder
DONE 5.5) put MNIST dataset in its own folder
DONE 5.51) Reorg data output of C files
DONE 6) create python script to run all versions

    DONE- collect total program time 
    DONE- collect total train time over 10,000 steps 
    DONE - collect full dataset forward pass time
    - collect cache misses
    DONE - will need to put a suppress output (--data_collection) argument into the programs so that python can parse their output
    DONE- graph each version's time in a bar chart

DONE 6.5) create a script to download MNIST dataset

6.6) connect to github so that I can download and install CBLAS and make a version with that
6.65) make script still work even when some things aren't downloaded
DONE 6.7) get rid of references to 41000/42000 and replace with 59000/60000
6.8) remove downloading behavior of matplotlib files


7) create a README.md file
    - hook visualization
    - usage instructions
    - discuss completely unoptimized C time
    - discuss methods used (compiler flags, flattening matrices, SIMD, threads, loop ordering, pointer arithmetic, etc)
    - show rest of graphics
    - discuss methods that didn't work (Strassen's, blocking) and why (matrices are very odd shapes)
    - discuss He initialization (trains to 90% accuracy in 1-2 epochs rather than 5-10 with uniform distr initialization)
    - Acknowledgements
        - fire matmul teacher thing
        - Samson Zhang video
        - maybe coffee before arch videos for general knowledge



