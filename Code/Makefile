# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

# Executable name
TARGET = CNN

# Source files
SRCS = CNN.c load_params.c Math.c

# Object files
OBJS = $(SRCS:.c=.o)

# Include headers (optional if all headers are in the same dir)
INCLUDES = -I.

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) -lm

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
