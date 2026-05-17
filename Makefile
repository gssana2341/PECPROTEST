# Makefile for Photonic project co-design and binding compilation

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -fopenmp -I./photonic/core -I./photonic/sim -I./photonic/training -I./photonic/interface -I./photonic/interface/hardware -I./photonic/lang -I./photonic/examples/mnist_small

# List of directories containing source files to compile
SRC_DIRS = photonic/core \
           photonic/sim \
           photonic/training \
           photonic/interface \
           photonic/interface/hardware \
           photonic/lang \
           photonic/security \
           photonic/examples/mnist_small

# Exclude generated code, test suites, and executable main entry points from core library
EXCLUDE_SRC = photonic/lang/mnist_compiled.c \
              photonic/lang/compiler_main.c \
              photonic/tests/test_analytic.c \
              photonic/tests/test_runner.c \
              photonic/examples/mnist_small/main.c

SRC_FILES = $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.c))
SRC = $(filter-out $(EXCLUDE_SRC), $(SRC_FILES))
OBJ = $(SRC:.c=.o)

all: libphotonic.a libphotonic.so

libphotonic.a: $(OBJ)
	ar rcs $@ $^

libphotonic.so: $(OBJ)
	$(CC) -shared -fopenmp -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) libphotonic.a libphotonic.so
