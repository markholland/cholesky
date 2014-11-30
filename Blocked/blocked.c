#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

inline int min(int a, int b) { return (a<b) ? a : b; }

int chol_blocked(double *A, int n, int b){

  int i,j,k,m;
  int info;
  const double one = 1.0;
  const double minusone = -1.0;
  double t1, t2;

  for(k = 0; k < n; k += b){
    m=min( n-k, b );
    dpotrf_("L", &m, &A[k*n+k], &n, &info);
    
    if( info < 0 ) {
      fprintf(stderr,"Error in the Cholesky decomposition \nInvalid argument");
      exit(-1);
    }
    if( info > 0 ) {
      fprintf(stderr,"Error in the Cholesky decomposition \nMatrix isn't positive definite");
      exit(-1);
    }

    for(i = k+b; i < n; i += b){
      m=min( n-i, b );  
      dtrsm_("R","L","T","N", &m, &b, &one, &A[k*n+k], &n, &A[i+k*n], &n);
    }
    
    for(i = k + b; i < n; i += b){
      m=min( n-i, b ); 
      for(j = k + b; j <= i - 1; j += b){
       dgemm_("N","T", &m, &b, &b, &minusone, &A[i+k*n], &n, &A[j+k*n], &n, &one, &A[j*n+i], &n);
    }
    dsyrk_("L","N",&m, &b, &minusone, &A[i+k*n], &n, &one, &A[i*n+i], &n);
    }
  }
  return 0;
}

int main( int argc, char *argv[] ) {

  int n, i, j, info, b;
  double *A;
  double t1, t2;

  if( argc<2 ) {
    fprintf(stderr,"usage: %s dimension block_size\n",argv[0]);
    exit(-1);
  }

  sscanf(argv[1],"%d",&n);
  sscanf(argv[2],"%d",&b);

  if( ( A = (double*) malloc(n*n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error reserving suficient memory for the matrix A\n");
    exit(-1);
  }

  // Fill the Matrix with random content
  for( j=0; j<n; j++ ) {
    for( i=j; i<n; i++ ) {
      A[i+j*n] = ((double) rand()) / RAND_MAX;
    }
    A[j+j*n] += n;
  }

  t1 = omp_get_wtime();
  chol_blocked(A,n,b);
  t2 = omp_get_wtime();

  printf("Time: %f\n", t2-t1);

  // Check result
  dpotrf_( "L", &n, A, &n, &info ); 
  
  if( info != 0 ) {
    fprintf(stderr,"Error = %d in the Cholesky decomposition. \n",info);
    exit(-1);
  }
  
  free(A);
}



