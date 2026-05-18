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
	rm -f $(OBJ) libphotonic.a libphotonic.so phoc

# ─── PhoLang Compiler ───────────────────────────────────
phoc: photonic/lang/compiler_main.c libphotonic.a
	$(CC) $(CFLAGS) $< -L. -lphotonic -lm -o $@

# ─── Demo: compile .pho แล้วรันเลย ──────────────────────
demo: phoc
	./phoc photonic/lang/mnist.pho /tmp/mnist_demo.c
	$(CC) $(CFLAGS) /tmp/mnist_demo.c \
		photonic/core/*.o \
		photonic/sim/*.o \
		photonic/training/*.o \
		photonic/examples/mnist_small/pooling.o \
		-lm -fopenmp -o /tmp/mnist_demo
	/tmp/mnist_demo

# ─── Test suite ──────────────────────────────────────────
test: libphotonic.a
	$(CC) $(CFLAGS) photonic/tests/test_runner.c \
		-L. -lphotonic -lm -o /tmp/test_runner
	/tmp/test_runner

# ─── Install phoc system-wide ────────────────────────────
install: phoc
	cp phoc /usr/local/bin/phoc
	@echo "phoc installed. Try: phoc your_network.pho output.c"

# ─── Help ────────────────────────────────────────────────
help:
	@echo "make          - build libphotonic.a and .so"
	@echo "make phoc     - build PhoLang compiler"
	@echo "make demo     - run MNIST demo end-to-end"
	@echo "make test     - run unit test suite"
	@echo "make install  - install phoc system-wide"
	@echo "make clean    - remove build artifacts"

.PHONY: all clean demo test install help
