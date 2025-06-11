# Makefile to build shared library libload.so from load.c and load.h

# Compiler and flags
CC = gcc
CFLAGS = -Wall -fPIC -O2
LDFLAGS = -shared -lm

# Source and target
SRC = load.c
TARGET = libload.so

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)
