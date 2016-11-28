CC=gcc -fPIC
FLAGS=-g -Wall -Wextra -pedantic -std=gnu99
OUT=libarray.a
AR=ar

all: $(OUT)


$(OUT): array.o array.h
	$(AR) rcs $@ $^ 

array.o : array.c array.h
	$(CC) $(FLAGS) -o $@ -c $<

clean:
	rm -f *.o *.a

rebuild: clean all

.PHONY: clean rebuild