# Simple Makefile for Photonic project

CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./photonic/core -I./photonic/sim -I./photonic/training -I./photonic/interface

SRC = $(wildcard photonic/**/*.c)
OBJ = $(SRC:.c=.o)

all: libphotonic.a

libphotonic.a: $(OBJ)
	ar rcs $@ $^

clean:
	rm -f $(OBJ) libphotonic.a
