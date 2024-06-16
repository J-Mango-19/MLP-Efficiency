CC= gcc
CFLAGS= -Wall -g -lm -std=gnu99 -O3
LD= gcc
LDFLAGS = -L. -lm 
TARGETS = mnist_nn clean_obj

all: $(TARGETS)

main.o: mnist_nn.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

utils.o: utilities.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

matrix_ops.o: matrix_ops.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

mnist_nn: main.o utils.o matrix_ops.o
	$(LD) $(LDFLAGS) -o $@ $^

clean:
	@rm -f *.o mnist_nn

clean_obj:
	@rm -f *.o

