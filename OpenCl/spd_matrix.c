#include "spd_matrix.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float **dmatrix(long nrl, long nrh, long ncl, long nch)
{
	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	float **m;

	m = (float**) malloc((size_t)((nrow + 1) * sizeof(float*)));
	m += 1;
	m -= nrl;

	m[nrl] = (float*) malloc((size_t)((nrow * ncol + 1) * sizeof(float)));
	m[nrl] += 1;
	m[nrl] -= ncl;

	for(i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;
    int j;
    for (i = 1; i <= nrh; i++)
        for (j = 1; j <= nch; j++)
            m[i][j] = 0.0;
	return m;
}

float random_double(float fMin, float fMax)
{
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void print_matrix(float** A, int dimension)
{
	int i,j;

	for(i = 1; i <= dimension; i++)
    {
    	for(j = 1; j <= dimension; j++)
    	{
    		printf("%20.16lf\t\t\t",A[i][j]);
    	}
      	printf("\n");
    }
}

void print_matrix_to_file(float** A, int dimension)
{
    FILE * pFile;
    int n;

    pFile = fopen ("bin/Debug/output","w");
    int i,j;

	for(i = 1; i <= dimension; i++)
    {
    	for(j = 1; j <= dimension; j++)
    	{
    		fprintf(pFile, "%20.16lf\t\t\t",A[i][j]);
    	}
      	fprintf(pFile, "\n");
    }

    fclose (pFile);


}


float** generate_random_matrix(float** A, int dimension)
{
	int i,j;

	for (i = 1; i <= dimension; i++)
    {
        for (j = 1; j <= dimension; j++)
        {
            A[i][j] = random_double(0, 1);
            //printf("Result: %lf\n", A[i][j]);
        }
    }
	return A;
}

float** clone_matrix(float** A, int dimension)
{
	float** cloned_matrix = dmatrix(1, dimension, 1, dimension);
	int i,j;
	for(i = 1; i <= dimension; i++)
    {
    	for(j = 1; j <= dimension; j++)
    	{
        	cloned_matrix[i][j] = A[i][j];
        	//printf("%f, ",cloned_matrix[i][j]);
        }
    }
    return cloned_matrix;
}

float** transpose_matrix(float** A, int dimension)
{
	int i,j;
	float** A_t;
	A_t = clone_matrix(A, dimension);
	for(i = 1; i <= dimension; i++) //transponowanie macierzy
    {
    	for(j = 1; j <= dimension; j++)
        	A[j][i] = A_t[i][j];
    }
    return A;
}

float** construct_symetric_matrix(float** A, int dimension)
{
	float** temp;
	temp = clone_matrix(A, dimension);
	float** A_t;
	int i,j;
	A_t = transpose_matrix(temp, dimension);
	//print_matrix(A_t, dimension);
	for(i = 1; i <= dimension; i++) // A = A+A'
    	for(j = 1; j <= dimension; j++)
      		A[i][j] = A[i][j] + A_t[i][j];

    return A;
}

float** create_identity_matrix(int dimension) //n*I(n)
{
	float** nI = dmatrix(1, dimension, 1, dimension);
	int i,j;
	for(i = 1; i <= dimension; i++)
	{
    	for(j = 1; j <= dimension; j++)
    	{
    		if (i == j)
      			nI[i][j] = dimension;
      		else
      			nI[i][j] = 0;
      	}
    }
    return nI;
}

float** matrix_positive_definite(float** A, int dimension)
{
	float** nI;
	int i,j;
	nI = create_identity_matrix(dimension);

	//print_matrix(I,dimension);
	for(i = 1; i <= dimension; i++) // A = A + n*I(n);
    	for(j = 1; j <= dimension; j++)
      		A[i][j] = A[i][j] + nI[i][j];

    return A;
}

float** create_lower_triangular(float **A, int dimension)
{
    float p, temp;
    int i, j, k;

    for(i = 1; i <= dimension; i++)
    {
        for(j = 2; j <= dimension - i; j++)
        {
            if(A[i + j][i] / A[i][i])
            {
                p = A[i + j][i] / A[i][i];
                for(k = 1; k <= dimension - i; k++)
                {
                    temp = A[i][i + k] * p;
                    A[i + j][i + k] -= temp;
                }
            }
        }
    }
    return A;
}

float** multiply(float **L, float **L_t, float **A, int dimension)
{
    int i, j, k;

    for (i = 1; i <= dimension; i++)
    {
        for(j = 1; j <= dimension; j++)
        {
            A[i][j] = 0;
            for(k = 1; k <= dimension; k++)
                A[i][j] += L[i][k] * L_t[k][j];
        }
    }

return A;
}

float frobenius_norm(float** L, int dimension)
{
    int i, j;
    float norm = 0.0;

    for (i = 1; i <= dimension; i++)
    {
        for (j = 1; j <= dimension; j++)
            norm += L[i][j] * L[i][j];
    }

return sqrt(norm);
}

float* convert_to_array(float** A, int dimension)
{
float *p = malloc(sizeof(float) * dimension * dimension);
int i, j;
int k = 0;

    for (i = 1; i <= dimension; i++)
        for(j = 1; j <= dimension; j++)
        {
            p[k] = A[i][j];
            k++;
        }

return p;
}

float** convert_to_matrix(float *A, int dimension)
{
float** P = dmatrix(1, dimension, 1, dimension);
int k;
int i = 1;
int j = 1;

    for (k = 0; k < dimension * dimension; k++)
    {
        P[i][j] = A[k];
        if (((k + 1) % dimension) != 0) j++;
        else
        {
            i++;
            j = 1;
        }
    }

return P;
}









