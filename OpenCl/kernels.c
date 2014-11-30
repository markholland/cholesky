#define A(x,y) A[x*dimension + y]
#define L(x,y) L[x*dimension + y]

__kernel void choldc_gpu(__global float* A, __global float* L, const unsigned int dimension)
{
int x = get_global_id(0);
int i, k, j;
float sum;

    //zeby workery ktore wykraczaja poza obszar macierzy nic nie kombinowaly
    if (x < dimension)
    {
//        //to robia wszyscy sekwencyjnie
        for (k = 0; k < dimension; k++)
        {
            //tu sie nic nie zrownolegli
            sum = A(k,k);
            for (j = 0; j < k - 1; j++) sum = sum - L(k,j) * L(k,j);
            L(k,k) = sqrt(sum);
            //za to tutaj mozna uzyc workerow wiekszych od aktualnego k
            //do przeliczenia rownolegle wszystkich wierszy pod elementem na diagonali
            if (x > k)
            {
                //kazdy worker bierze po jednym i ( po jednym wierszu )
                i = x + 1;
                sum = A(i,k);
                for (j = 0; j < k - 1; j++) sum = sum - L(i,j) * L(k,j);
                L(i,k) = sum / L(k,k);
            }
        }
    }

}

__kernel void choldc2_gpu(__global float* A, __global float* L, const unsigned int dimension)
{
int x = get_global_id(0);
int i, k, j;

    //zeby workery ktore wykraczaja poza obszar macierzy nic nie kombinowaly
    if (x < dimension)
    {
        //to robia wszyscy sekwencyjnie
        for (k = 0; k < dimension - 1; k++)
        {
            L(k,k) = sqrt(A(k,k));
            //wszystkie workery ktore przelecialy powyzej k moga to zrobic jednoczesnie
            if (x >= k)
            {
                i = x + 1;
                L(i,k) = A(i,k) / L(k,k);
            }
            //dobra tutaj trzeba zaczekac zeby wszystkie workery zrobily co trzeba
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            //i to tez
            if (x >= k)
            {
                j = x + 1;
                for (i = j; i < dimension; i++) A(i,j) = A(i,j) - L(i,k) * L(j,k);
            }
        }
        if (x == dimension - 1)
            L(x,x) = sqrt(A(x,x));
    }
}
