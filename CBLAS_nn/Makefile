CC= gcc
CFLAGS= -Wall -fopenmp -g -lm -std=gnu99 -mavx -march=native -O3 -march=native 
LD= gcc
LDFLAGS = -L. -lm -lopenblas #-fopenmp
TARGETS = mnist_nn clean_obj

all: $(TARGETS)

main.o: main.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

net.o: neural_network.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

utils.o: utils.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

matrix_ops.o: matrix_operations.c mnist.h 
	$(CC) $(CFLAGS) -o $@ -c $<

mnist_nn: main.o net.o utils.o matrix_ops.o
	$(LD) $^ $(LDFLAGS) -o $@

clean:
	@rm -f *.o mnist_nn

clean_obj:
	@rm -f *.o

