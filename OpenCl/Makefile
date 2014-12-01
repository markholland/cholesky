OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
	OPTIONS += -lm
endif

main: main.c
	gcc -fopenmp -Wall -g main.c cholesky.c spd_matrix.c -o main -L /opt/AMDAPPSDK-2.9-1/lib/x86_64/ $(OPTIONS)

clean:
	rm -rf main
	rm -rf *.o
