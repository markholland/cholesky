#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void chol_escalar(double *A, int n){

  int k, i, j, l;
  double c;

  for(k=0; k<n; k++){

    c=sqrt(A[k*n+k]);
    A[k*n+k]=c;

    for(i=k+1; i<=n; i++) {
      #pragma omp task
      A[i+k*n]=A[i+k*n]/c;
    }

    #pragma omp taskwait
    
    for (l=k+1; l<=n; l++) {
      for (j=k+1; j<=l-1; j++) {
        #pragma omp task
	      A[l+j*n]=A[l+j*n]-A[l+k*n]*A[j+k*n];
      }
      #pragma omp task
      A[l*n+l]=A[l*n+l]-A[l+k*n]*A[l+k*n];
    }
  }
}

int main( int argc, char *argv[] ) {
  int n, i, j, info;
  double *A;

  double t1, t2;

  if( argc<2 ) {
    fprintf(stderr,"usage: %s n\n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);

  if( ( A = (double*) malloc(n*n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error reserving suficient memory for the matrix A\n");
    exit(-1);
  }

  // Fill the Matrix with random content
  for( j=0; j<n; j++ ) {
    for( i=j; i<n; i++ ) {
      A[j*n+i] = ((double) rand()) / RAND_MAX;
    }
    A[j*n+j] += n;
  }

  t1 = omp_get_wtime();
  #pragma omp parallel
  #pragma omp single
  chol_escalar(A,n);
  t2 = omp_get_wtime();
  
  printf("\n");
  printf("Time: %f\n\n", t2-t1);

  // Check result
  dpotrf_( "L", &n, A, &n, &info ); 
  
  if( info != 0 ) {
    fprintf(stderr,"Error = %d in the Cholesky decomposition. \n",info);
    exit(-1);
  }

  free(A);
}