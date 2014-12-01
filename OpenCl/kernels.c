#define A(x,y) A[x*dimension + y]
#define L(x,y) L[x*dimension + y]

__kernel void choldc_gpu(__global float* A, __global float* L, const unsigned int dimension)
{
int x = get_global_id(0);
int i, k, j;
float sum;

    if (x < dimension)
    {
        for (k = 0; k < dimension; k++)
        {
            sum = A(k,k);
            for (j = 0; j < k - 1; j++) {
                sum = sum - L(k,j) * L(k,j);
            }
            L(k,k) = sqrt(sum);
            if (x > k)
            {
                i = x + 1;
                sum = A(i,k);
                for (j = 0; j < k - 1; j++) sum = sum - L(i,j) * L(k,j);
                L(i,k) = sum / L(k,k);
            }
        }
    }

}
