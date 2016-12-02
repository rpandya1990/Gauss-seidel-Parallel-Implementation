We first start by analyzing the problem which is also popularly known as the Gauss-Siedel or stencil operation.

### Identifying Dependencies

  - Each row elements depends on the element to the left
  - Each column depends on the previous column

### Approaches
We will discuss and implement two ways of solving the problem and then discuss the tradeoffs of each approach:
  - Wave Solver
  - Red and Black Solver

#### Wave Solver

We can observe from the order of computaions for different elements of the matrix that a wave pattern exists. All elements in a single diagonal can be computed parallely but the computation of diagonals is sequential as next diagonal depends on the values from previous values. 

See the below figure to get the idea.


![Gauss Siedel Waves](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%204.png)


> Blue lines represent the diagonals
> Elements in the diagonal can be computed parallely
> Each diagonal should wait on the completion for diagonals on it's left
 
 ```
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
    }
```

The outer loop iterates over the diagonals sequentially and the inner loop calculates the elements on a particular diagonal.

We achieve the parallelism with :
```
#pragma omp parallel for num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
```
We can specify the number of threads to be used for computation and workload is equally distributed among threads. We also use the```reduction(+:diff)``` directive which instructs threads to accumalate the difference locally and then combine at the end when thread operations are done. This improves the parallelism.

Advantages: 

- Easy and Effective parallelism 
- Scaling: As the workload is distributed equally, it scales well
- Correctness: Exactly same result as the serial approach

Disadvantages:

- Frequent synchronizations between diagonals equal to " 2N - 1 "
- Not much parallelism in initial and final diagonals

### Red and black coloring approach

We slightly change the algorithm with the domain knowledge that an approximate solution is acceptable for applications using Gauss Siedel.

Key Ideas:
- Change the order in which cells are updated
- New algorithm converges in same(approximate) number of iterations
- Change is acceptable for applications utilizing the Gauss Siedel method

We split the elements into red and black color scheme such that no red element depends on black and vice versa
![Red Black color scheme](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%205.png)

Firstly, red cells are computed in parallely and then the black cells are computed. We repeat the process until we acheive the desired convergence.
```
#pragma omp parallel num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
    {
      #pragma omp for
      for (i = 1; i <= n; ++i)
      {    
        for (j = 1; j <= n; ++j)
        { 
          if ((i + j) % 2 == 1)
          {
            // Computation for Red Cells
          }
        }
      }
      #pragma omp barrier
    }
    #pragma omp parallel num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
    {
      #pragma omp for
      for (i = 1; i <= n; ++i)
      {    
        for (j = 1; j <= n; ++j)
        { 
          if ((i + j) % 2 == 0)
          {
            // Computation for Black Cells
          }
        }
      }
      #pragma omp barrier
```
Advantages:

- Faster compared to wave approach and for smaller dataset as well
- Uniform parallelism

Disadvantages:
- Approximate solution

### Analysis

![General](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%206.png)

![Correctness](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%207.png)

### Charts

![SerialvsParallel](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%201.png)

![StrongScaling](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%202.png)

![Correctness](https://raw.githubusercontent.com/rpandya1990/Gauss-seidel-Parallel-Implementation/master/images/Image%203.png)

### Observation

- Red black performs very well for all range of Grid size
- Red black is the fastest
- Wave Solver is completely accurate and the result exactly matches the Serial Implementation
- Approximate solution given by red black solver is acceptable

### Running the code

To run the Parallel Wave Solver in ELF:

```sh
$ g++ -o wave_solver parallel_omp_wave.cpp -fopenmp
$ ./wave_solver
```

To run the Parallel Red and Black Solver in ELF:

```sh
$ g++ -o rb_solver parallel_rb.cpp -fopenmp
$ ./rb_solver
```

