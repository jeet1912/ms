#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define COLOR 1<<10  /* 1024 */
#define MAXDIM 1<<12 /* 4096 */
#define ROOT 0

int mat_mult(double *A, double *B, double *C, int n, int n_local);
void init_data(double *data, int data_size);
int check_result(double *C, double *D, int n);
int is_power_of_two(int x);

int main(int argc, char *argv[]) {
    int n = 64,n_sq, flag, my_work;
    int my_rank, num_procs = 1;
    double *A, *B, *C, *D;	/* D is for serial computation */
    int elms_to_comm, row_offset, elem_offset;
    double start_time, end_time, elapsed;

    MPI_Comm world = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(world, &my_rank);
    MPI_Comm_size(world, &num_procs);

    /* Read n from command line */
    if (argc > 1) {
        n = atoi(argv[1]);  // ASCII to integer
        if (n>MAXDIM) n = MAXDIM;
    }

    if (!is_power_of_two(n) || !is_power_of_two(num_procs)) {
        if (my_rank == ROOT) {
            printf("Error: n and num_procs must be powers of two.\n");
        }
        MPI_Abort(world, 1);
    }

    if (n % num_procs != 0) {
        if (my_rank == ROOT) {
            printf("Error: n must be divisible by num_procs (row split).\n");
        }
        MPI_Abort(world, 1);
    }


    n_sq = n * n; // total number of elements in a matrix 
    my_work  = n/num_procs; // rows per process
    elms_to_comm = my_work * n;  // elements to send to each process

    /* Global offsets for this rank */
    row_offset  = my_rank * my_work;  /* starting global row */
    elem_offset = row_offset * n;     /* starting element index */

    A = (double *) malloc(sizeof(double) * n_sq);
    B = (double *) malloc(sizeof(double) * n_sq);
    C = (double *) malloc(sizeof(double) * n_sq);
    D = (double *) malloc(sizeof(double) * n_sq);

    if (!A || !B || !C || !D) {
        printf("Rank %d: malloc failed.\n", my_rank);
        MPI_Abort(world, 1);
    }
    if (my_rank == ROOT) {
        printf("pid=%d: num_procs=%d n=%d my_work=%d\n", my_rank, num_procs, n, my_work);
        srand(12345); // seed for random number generator
        init_data(A, n_sq);
        init_data(B, n_sq);
    }

    start_time = MPI_Wtime();

    /* Scatter rows of A to all processes */
    MPI_Scatter(A, elms_to_comm, MPI_DOUBLE, &A[elem_offset], elms_to_comm, MPI_DOUBLE, ROOT, world);

    /* Broadcast B to all processes */
    MPI_Bcast(B, n_sq, MPI_DOUBLE, ROOT, world);

    /* Each process computes its own mat mult */
    mat_mult(&A[elem_offset], B, &C[elem_offset], n, my_work);

    /* Gather all results to C on root */
    MPI_Gather(&C[elem_offset], elms_to_comm, MPI_DOUBLE, C, elms_to_comm, MPI_DOUBLE, ROOT, world);

    /* Check the result */
    if (my_rank == ROOT) {
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;

        /* Local computation for comparison: results in D */
        mat_mult(A, B, D, n, n);

        flag = check_result(C,D,n);
        if (flag) printf("Test: FAILED\n");
        else {
            printf("Test: PASSED\n");
            printf("Total time %d: %f seconds.\n", my_rank, elapsed);
        }
    }
    free(A); free(B); free(C); free(D);
    MPI_Finalize();
    return 0;
}


int mat_mult(double *a, double *b, double *c, int n, int my_work) {
  int i, j, k;
  double sum=0.0;
  for (i=0; i<my_work; i++) {
    for (j=0; j<n; j++) {
      sum=0;
      for (k=0; k<n; k++){
        sum += a[i*n + k] * b[k*n + j];
      }
      c[i*n + j] = sum;
    }
  }
  return 0;
}


/* Initialize an array with random data */
void init_data(double *data, int data_size) {
  for (int i = 0; i < data_size; i++)
    data[i] = rand() & 0xf; /* numbers between 0 and 15 */
}


/* Compare two matrices C and D */
int check_result(double *C, double *D, int n){
  int i,j,flag=0;
  double *cp,*dp;

  cp = C;
  dp = D;

  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      if (*cp++ != *dp++) {
        printf("ERROR: C[%d][%d]=%f != D[%d][%d]=%f\n", i, j, C[i*n + j], i, j, D[i*n + j]);
        flag = 1;
        return flag;
      }
    }
  }
  return flag;
}

/* Check if x is a power of two */
int is_power_of_two(int x) {
  return (x > 0) && ((x & (x - 1)) == 0);
}