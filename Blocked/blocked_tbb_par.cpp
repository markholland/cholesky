#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
//#include "tbb/blocked_range.h"

using namespace tbb;

extern "C" void dpotrf_(char *, int *, double*, int*, int *);
extern "C" void dtrsm_(char *, char *, char *, char *, int *, int *, const double *, double *, int *, double *, int *);
extern "C" void dgemm_(char *, char *, int *, int *, int *, const double *, double *, int *, double *, int *, const double *, double *, int *);
extern "C" void dsyrk_(char *, char *, int *, int *, const double *, double *, int *, const double *, double *, int *);

inline int min(int a, int b) { return (a<b) ? a : b; }

int chol_bloques(double *A, int n, int b){

  int i,j,k,m;
  int info;
  const double one = 1.0;
  const double minusone = -1.0;

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

    parallel_for(k+b, n-1, b, [&m, &b, &one, &A, &k, &n, &i](size_t ind){
      i = static_cast<int>(ind);
      m=min( n-ind, b );  
      dtrsm_("R","L","T","N", &m, &b, &one, &A[k*n+k], &n, &A[ind+k*n], &n);
    });

    for(i = k + b; i < n; i += b){
      m=min( n-i, b ); 
      parallel_for(k+b, i-1 , b, [&m, &b, &i, &k,&minusone, &A, &n, &one](size_t ind){
        dgemm_("N","T", &m, &b, &b, &minusone, &A[i+k*n], &n, &A[ind+k*n], &n, &one, &A[ind*n+i], &n);
      });
    dsyrk_("L","N",&m, &b, &minusone, &A[i+k*n], &n, &one, &A[i*n+i], &n);
    }
  }

  return 0;
}

int main(int argc, char *argv[])
{
  int n, i, j, info, b, hilos;
  double *A;

  double t, t1, t2;

  if( argc<3 ) {
    fprintf(stderr,"usage: %s n b\n",argv[0]);
    exit(-1);
  } 

  sscanf(argv[1],"%d",&hilos);
  sscanf(argv[2],"%d",&n);
  sscanf(argv[3],"%d",&b);

  task_scheduler_init init(hilos);

  
  if( ( A = (double*) malloc(n*n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para la matriz A\n");
    exit(-1);
  }
  for( j=0; j<n; j++ ) {
    for( i=j; i<n; i++ ) {
      A[j*n+i] = ((double) rand()) / RAND_MAX;
    }
    A[j*n+j] += n;
  }

  t1 = omp_get_wtime();

  chol_bloques(A,n,b);

  t2 = omp_get_wtime();
  t = t2 - t1;
  //fin toma tiempos

  /*printf("\n");

  for(i=0; i<n; i++){
    printf("\n");
    for(j=0; j<n; j++){
      printf("%10.3lf",A[j*n+i]);
    }
  }

  printf("\n");
  
  printf("\n\n");*/
  
  printf("Tiempo de computo: %f\n\n", t);

  dpotrf_( "L", &n, A, &n, &info ); 
  //L(triangular inf), n(dimension), A(dir. primer elemento), info(resultado).
  if( info != 0 ) {
    fprintf(stderr,"Error = %d en la descomposicion de Cholesky \n",info);
    exit(-1);
  }
  free(A);
}
