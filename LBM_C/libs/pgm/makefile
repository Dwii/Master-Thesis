CC=gcc -fPIC
FLAGS=-g -Wall -Wextra -pedantic -std=gnu99
OUT=libpgm.a

all: $(OUT)

$(OUT): pgm.o pgm.h
	$(AR) rcs $@ $^ 

array.o : pgm.c pgm.h
	$(CC) $(FLAGS) -o $@ -c $<

clean:
	rm -f *.o *.a

rebuild: clean all

.PHONY: clean rebuild