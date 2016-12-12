CC=gcc
OUT=lbmFlowAroundCylinder

FLAGS=-g -Wall -Wextra -pedantic -std=gnu99

LIBS=$(addprefix ../libs/,pgm array)
I_FLAGS=$(addprefix -I,$(LIBS))
L_FLAGS=$(addprefix -L,$(LIBS))

IMG_PATH=./images

all: $(OUT)

run: all $(IMG_PATH)
	./lbmFlowAroundCylinder $(IMG_PATH)

$(IMG_PATH): 
	-mkdir -p $@

$(OUT): lbmFlowAroundCylinder.o
	$(CC) $(FLAGS) $< -o $@ -Bdynamic -larray -lpgm $(L_FLAGS)

lbmFlowAroundCylinder.o : lbmFlowAroundCylinder.c
	$(CC) $(FLAGS) -o $@ -c $< $(I_FLAGS)

clean:
	rm -rf *.o *.so $(OUT) $(IMG_PATH)

rebuild: clean all

.PHONY: clean rebuild