#include "cholesky.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


float** choldc(float **A, float **L, int dimension)
{
    int i,j,k;
    float c;
    float sum;
    clock_t begin, end;

    begin = clock();
    for (k = 1; k <= dimension; k++)
    {
        for(k=1; k<=dimension; k++) {
            c=sqrt(A[k][k]);
            A[k][k]=c;

            for(i=k+1; i<=dimension; i++) {
                A[i][k]=A[i][k]/c;
            }

        for (i=k+1; i<=dimension; i++) {
            for (j=k+1; j<=i-1; j++) {
                A[i][j]=A[i][j]-A[i][k]*A[j][k];
            }
            A[i][i]=A[i][i]-A[i][k]*A[i][k];
            }
        }   
    }

    end = clock();
    printf("Time \tCPU: % 20.16lf\n", (float)(end - begin) / CLOCKS_PER_SEC);

return A;
}
