#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

#define N 201
#define numThreads 5

double ** A, **B, diff;
double **visited;

typedef struct {
    int start_row;
    long threadId;
} WorkerArgs;
WorkerArgs args[numThreads];

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
   int convergence=FALSE;
   double diff, tmp;
   int i,j, iters=0;
   int for_iters;


   for (for_iters=1;for_iters<2;for_iters++) 
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

     if (diff/((double)N*(double)N) < Tolerance)
       convergence=TRUE;

    } /*for*/
}

void compute(int i, int j)
{
  // printf("Computing %d, %d\n", i, j);
  double tmp;
  A[i][j] = tmp;
  A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
  visited[i][j] = 1;
  diff += fabs(A[i][j] - tmp);
}

void  *workerThreadStart(void *id) {
    long id1 = (long)id;
    int i, j;
    // printf("Thread: %ld, Start: %d/n", args[id1].threadId, args[id1].start_row);
    for (i = args[id1].start_row; i < N; i=i+numThreads)
    {
      for (j = 1; j < N; ++j)
      { 
        while(1)
        { 
          if (visited[i-1][j] == 1 && visited[i][j-1] == 1)
          {
            compute(i, j);
            break;
          }
        }
      }
    }
    return NULL;
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
  printf("Count: %d\n", count);
  return flag;
}

int main(int argc, char * argv[])
{
   int i, k;
   long j;
   long t_start,t_end;
   double time;
   pthread_t workers[numThreads];
   // display_visited(visited);
   A = malloc((N+2) * sizeof(double *));
   B = malloc((N+2) * sizeof(double *));
   visited = malloc((N+2) * sizeof(double *));
   for (i=0; i<N+2; i++) 
   {
	   A[i] = malloc((N+2) * sizeof(double)); 
     B[i] = malloc((N+2) * sizeof(double));
     visited[i] = malloc((N+2) * sizeof(double));
   }

   initialize(visited, N);
   initialize(B, N);

   t_start=usecs();
   solve(B, N);
   t_end=usecs();

   time = ((double)(t_end-t_start))/1000000;
   printf("Computation time for serial = %f\n", time);
   
   // display(B, N);

   initialize(A, N);

   for (i = 0; i < numThreads; ++i)
    {
      args[i].start_row = i + 1;
      args[i].threadId = i;
    }
   t_start=usecs();
   for (j=0; j<numThreads; j++)
      pthread_create(&workers[j], NULL, workerThreadStart, (void *)args[j].threadId);
   for (k=0; k<numThreads; k++)
      pthread_join(workers[k], NULL);
   t_end=usecs();

   time = ((double)(t_end-t_start))/1000000;
   printf("Computation time for parallel = %f\n", time);

   // display(A, N);

   printf("Parallel is correct: %d\n", verify());

}