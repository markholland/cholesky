#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void chol_escalar(double *A, int n){

  int k, i, j;
  double c;
  
  for(k=0; k<n; k++) {
    c=sqrt(A[k+k*n]);
    A[k+k*n]=c;

    for(i=k+1; i<n; i++) {
      A[i+k*n]=A[i+k*n]/c;
    }

    for (i=k+1; i<n; i++) {
      for (j=k+1; j<=i-1; j++) {
	       A[i+j*n]=A[i+j*n]-A[i+k*n]*A[j+k*n];
      }
      A[i+i*n]=A[i+i*n]-A[i+k*n]*A[i+k*n];
    }
  }   
}

int main( int argc, char *argv[] ) {

  int n, i, j, info, b;
  double *A;
  double t1, t2;

  if( argc<2 ) {
    fprintf(stderr,"usage: %s dimension\n",argv[0]);
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
      A[i+j*n] = ((double) rand()) / RAND_MAX;
    }
    A[j+j*n] += n;
  }

  t1 = omp_get_wtime();
  chol_escalar(A,n);
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

