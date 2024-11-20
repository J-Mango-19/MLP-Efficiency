CC= gcc
CFLAGS= -Wall -lm -mavx -march=native -Ofast
LD= gcc
# Add the directory where OpenBLAS is installed (assuming ~/local/openblas)
OPENBLAS_PATH=$(HOME)/local/openblas
LDFLAGS=-L$(OPENBLAS_PATH)/lib -lopenblas -lm -lpthread
# Also add the include path for OpenBLAS headers, if needed
CFLAGS += -I$(OPENBLAS_PATH)/include

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

