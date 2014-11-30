#ifndef SPD_MATRIX
#define SPD_MATRIX

float **dmatrix();
float random_double(float fMin, float fMax);
void print_matrix(float** A, int dimension);
void print_matrix_to_file(float** A, int dimension);
float** generate_random_matrix(float** A, int dimension);
float** clone_matrix(float** A, int dimension);
float** transpose_matrix(float** A, int dimension);
float** construct_symetric_matrix(float** A, int dimension);
float** create_identity_matrix(int dimension);
float** matrix_positive_definite(float** A, int dimension);
float** create_lower_triangular(float **A, int dimension);
float** multiply(float **L, float **L_t, float **A, int dimension);
float frobenius_norm(float** L, int dimension);

float* convert_to_array(float **A, int dimension);
float** convert_to_matrix(float *A, int dimension);
#endif // SPD_MATRIX
