#include "cholesky.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

float** choldc(float **A, float **L, int dimension)
{
    int i,j,k;
    float sum;
    clock_t begin, end;

    begin = clock();
    for (k = 1; k <= dimension; k++)
    {
        sum = A[k][k];
        for (j = 1; j <= k - 1; j++) sum = sum - (L[k][j] * L[k][j]);
        L[k][k] = sqrt(sum);
        for (i = k + 1; i <= dimension; i++)
        {
            sum = A[i][k];
            for (j = 1; j <= k - 1; j++) sum = sum - L[i][j] * L[k][j];
            L[i][k] = sum / L[k][k];
        }
    }
    end = clock();
    printf("Time \tCPU: % 20.16lf\n", (float)(end - begin) / CLOCKS_PER_SEC);

return L;
}


