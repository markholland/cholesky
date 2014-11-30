OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
	OPTIONS += -lm
endif

main: main.c
	gcc -fopenmp -Wall -g main.c cholesky.c spd_matrix.c -o main $(OPTIONS)

clean:
	rm -rf main
	rm -rf *.o
