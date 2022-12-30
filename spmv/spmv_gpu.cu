#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)                                                                         \
{                                                                                           \
    const cudaError_t err = call;                                                           \
    if(err != cudaSuccess) {                                                                \
        printf("%s in %s at line %d\n", cudaGetErrorName(err), __FILE__, __LINE__);         \
        exit(EXIT_FAILURE);                                                                 \
    }                                                                                       \
}                                                                                           

#define CHECK_KERNEL_CALL()                                                                 \
{                                                                                           \
    const cudaError_t err = cudaGetLastError();                                             \
    if(err != cudaSuccess) {                                                                \
        printf("%s in %s at line %d\n", cudaGetErrorName(err), __FILE__, __LINE__);         \
        exit(EXIT_FAILURE);                                                                 \
    }                                                                                       \
}                                                                                           


double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    printf("Got the number of rows etc...\n");
    printf("%d rows, %d cols, %d vals\n", *num_rows, *num_cols, *num_vals);
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    float *values_t = (float *) malloc(*num_vals * sizeof(float));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    printf("row_ptr array ready\n");

    // Set the file position to the beginning of the file
    rewind(file);
    

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    printf("col_ind array ready\n");

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }

    printf("values saved in memory\n");
    
    fclose(file);
    

    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;

    printf("Set up was completed!\n\n");
}

// CPU implementation of SPMV using CSR, DO NOT CHANGE THIS
void spmv_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    for (int i = 0; i < num_rows; i++) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        
        y[i] = dotProduct;
    }
}

void check_results(const float *a1, const float *a2, const int rows) {
    for(int i = 0; i < rows; i++) {
        if(abs(a1[i] - a2[i]) > 0.003) {
            printf("should be %lf, got %lf at index %d\n", a1[i], a2[i], i);
            return;
        }
    }
    printf("COMPUTATION WAS CORRECT\n");
}


__global__ void spmv_csr_gpu(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *d_x, float *y, const int num_threads) {
    
    int single_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    for(int index = single_thread_id; index < num_rows; index = index + num_threads) {
        
        float dot_product = 0;
        int row_start = row_ptr[index];
        int row_end = row_ptr[index+1];

        for(int j = row_start; j < row_end; j++) {
            dot_product += values[j] * d_x[col_ind[j]];
        }

        y[index] = dot_product;
    }
}



int main(int argc, const char * argv[]) {

    if (argc != 3) {
        printf("Usage: ./exec matrix_file num_threads");
        return 0;
    }
    
    // Variables for cpu

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    
    int *d_row_ptr, *d_col_ind;
    float *d_values;
    float *d_x;
    float *d_y;
    
    const char *filename = argv[1];
    const int num_threads = atoi(argv[2]);

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;
    
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y_sw = (float *) malloc(num_rows * sizeof(float));
    float *result = (float *) malloc(num_rows * sizeof(float));

    // Allocating gpu memory for the various arrays
    printf("Allocating GPU memory\n");    

    CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&d_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&d_y, (num_rows + 1) * sizeof(float)));

    // Generate a random vector

    srand(time(NULL));

    for (int i = 0; i < num_rows; i++) {
        x[i] = (rand()%100)/(rand()%100+1); //the number we use to divide cannot be 0, that's the reason of the +1
    }
    
    CHECK(cudaMalloc(&d_x, (num_rows + 1) * sizeof(float)));

    // Copying data from cpu to gpu

    printf("Copying data from CPU to GPU\n");
    CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x, (num_rows + 1) * sizeof(float), cudaMemcpyHostToDevice));

    // Compute in GPU

    printf("GPU computation\n");
    dim3 threadsPerBlock(256);
    dim3 numBlocks(num_threads/256);

    start_gpu = get_time();
    spmv_csr_gpu<<<numBlocks, threadsPerBlock>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        d_x,
        d_y,
        num_threads
      );
    CHECK_KERNEL_CALL();
    cudaDeviceSynchronize();

    CHECK(cudaMemcpy(result, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    end_gpu = get_time();
    
    // Compute in sw

    printf("CPU computation\n");

    start_cpu = get_time();
    spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
    end_cpu = get_time();

    // Verifying result
    check_results(y_sw, result, num_rows);

    // Print time
    printf("SPMV Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SPMV Time GPU: %.10lf\n", end_gpu - start_gpu);
    
    // Free    
    free(row_ptr);
    free(col_ind);
    free(values);
    free(y_sw);
    free(result);

    // GPU memory free
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
