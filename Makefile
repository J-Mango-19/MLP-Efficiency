CC= gcc
CFLAGS= -Wall -g -lm -std=gnu99 -O3
LD= gcc
LDFLAGS = -L. -lm 
TARGETS = mnist clean_obj

all: $(TARGETS)


mnist.o: mnist.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

input.o: input.c mnist.h
	$(CC) $(CFLAGS) -o $@ -c $<

mnist: mnist.o input.o
	$(LD) $(LDFLAGS) -o $@ $^


clean:
	@rm -f *.o mnist

clean_obj:
	@rm -f *.o

