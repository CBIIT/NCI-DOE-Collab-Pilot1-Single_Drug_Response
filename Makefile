OBJS = options.o mpi_util.o io.o missing.o random_forest.o

CC = mpic++
cc = mpicc

PROFILE = #-pg -g
OPENMP = -fopenmp

FLAGS = $(PROFILE) -O3 -Wall $(OPENMP) -std=c++0x

INC = -I.
LIBS = -lm -lz

.SUFFIXES : .o .cpp .c
.cpp.o:
	$(CC) $(FLAGS) $(PROFILE) $(INC) -c $<
.c.o:
	$(cc) $(FLAGS) $(PROFILE) $(INC) -c $<

all: xTx

xTx : $(OBJS)  xTx.o
	$(CC) $(PROFILE) -o xTx xTx.o $(OBJS) $(LIBS) $(OPENMP)

clean:
	-rm -f *.o


