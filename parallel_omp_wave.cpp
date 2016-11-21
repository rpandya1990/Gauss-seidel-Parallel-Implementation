#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

#define N 8001
#define THREAD_COUNT 16

double ** A, **B;

void display(double **V, int n)
{
  for (int i = 0; i < n+1; ++i)
    {
        for (int k = 0; k < n+1; ++k)
        {
            printf("%f ", V[i][k]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialize (double **A, int n)
{
   int i,j;

   for (j=0;j<n+1;j++){
     A[0][j]=1.0;
   }
   for (i=1;i<n+1;i++){
      A[i][0]=1.0;
      for (j=1;j<n+1;j++) A[i][j]=0.0;
   }
}

void solve(double **B, int n)
{
   printf("\n\n-----------------------Serial Solver-----------------------\n\n\n");
   int convergence=FALSE;
   double diff, tmp;
   int i,j, iters=0;
   int for_iters;

   for(for_iters = 1; for_iters < 21; for_iters++)
   { 
     diff = 0.0;

     for (i=1;i<n;i++)
     {
       for (j=1;j<n;j++)
       {
         tmp = B[i][j];
         B[i][j] = 0.2*(B[i][j] + B[i][j-1] + B[i-1][j] + B[i][j+1] + B[i+1][j]);
         diff += fabs(B[i][j] - tmp);
       }
     }
     iters++;
     printf("Difference after %3d iterations: %f\n", iters, diff);
     if (diff/((double)N*(double)N) < Tolerance)
     {
       printf("\nConvergence achieved after %d iterations....Now exiting\n\n", iters);
       return;
     }
    }
    printf("\n\nIteration LIMIT Reached...Exiting\n\n");
}

long usecs (void)
{
  struct timeval t;

  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}

int verify()
{
  int flag = 1, i, j;
  int count = 0;
  for (i = 1; i < N; ++i)
  {
    for (j = 1; j < N; ++j)
    {
      if (A[i][j] != B[i][j])
      {
        // printf("Element different: %d,%d\n", i, j);
        count++;
        flag = 0;
      }
    }
  }
  // printf("  Different element in Matrices A and B: %d ", count);
  return flag;
}

void solve_parallel(double **A, int n)
{
  printf("\n\n-----------------------Parallel Wave Solver-----------------------\n\n\n");
  int for_iters;
  int diagonals = 2 * n - 1, i, j;
  int iters = 0;
  int convergence=FALSE;
  double tmp;
  double diff;
  for (for_iters = 1; for_iters < 21; ++for_iters)
  {
    diff = 0;
    for (int k = 1; k <= diagonals; ++k)
    { 
      // printf("Diagonal: %d\n", k);
      #pragma omp parallel for num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
      for (i = (k <= n ? 1 : (k - n + 1)); i <= k; i++)
        {   
          if(i <= n)
          {         
            j = k + 1 - i;
            // printf("Thread: %d on (%d, %d)\n", omp_get_thread_num(), i, j);
            tmp = A[i][j];
            A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
            diff += fabs(A[i][j] - tmp);
          }
        }  
        #pragma omp barrier   
    }
    iters++;
    printf("Difference after %3d iterations: %f\n", iters, diff);
     if (diff/((double)N*(double)N) < Tolerance)
     {
       printf("\nConvergence achieved after %d iterations....Now exiting\n\n", iters);
       return;
     }
  }
  printf("\n\nIteration LIMIT Reached...Exiting\n\n");
}

int main(int argc, char * argv[])
{  
   int i;
   long t_start,t_end;
   double time;
   A = new double *[N+2];
   B = new double *[N+2];
   for (i=0; i<N+2; i++) 
   {
     A[i] = new double[N+2];
     B[i] = new double[N+2];
   }

   initialize(B, N);

   t_start=usecs();
   solve(B, N);
   t_end=usecs();

   time = ((double)(t_end-t_start))/1000000;
   printf("Computation time for Serial approach(secs): %f\n\n", time);
   
   // display(B, N);

   initialize(A, N);

   t_start=usecs();
   solve_parallel(A, N - 1);
   t_end=usecs();

   time = ((double)(t_end-t_start))/1000000;
   printf("Computation time for Parallel approach(secs): %f\n\n", time);

   // display(A, N);

   printf("Validating the Parallel approach:\t");
   if (verify() == 1)
   {
    printf("SUCCESS\n\n");
   }
   else
   {
    printf("ERROR\n\n");
   }
}