CC 			=	nvcc
CFLAGS		=	-O3
PROG		=	spmv/spmv-gpu

all:$(PROG)

spmv/spmv-gpu: spmv/spmv_gpu.cu
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)
