// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <rocblas.h>
#include <hiprand.h>
#include <cmath>
#include "gputimer.hip.h"
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <sstream>

// Function to read data from file
float *read_from_file(std::string file_name)
{
	// lets get filesize
	struct stat results;
	if (stat(file_name.c_str(), &results) != 0)
	{
		// An error occurred
		std::cout << "ERROR: unable to get filesize" << std::endl;
		return NULL;
	}
	// The size of the file in bytes is in results.st_size
	// Lets allocate an array to contain the binary file
	float *data = (float *)malloc(results.st_size);

	// lets write it to binary file
	std::ifstream infile;

	// open a binary file
	infile.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

	//read data from file
	infile.read((char *)data, results.st_size);

	// close the file
	infile.close();

	return data;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A)
{
	// Create a pseudo-random number generator
	hiprandGenerator_t prng;
	hiprandCreateGenerator(&prng, HIPRAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	hiprandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// Fill the array with random numbers on the device
	hiprandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Randomization helpers
// adapted from https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/rocm-3.0/clients/include/rocblas_init.hpp#L42
void fill_sin(float *A, size_t nr_rows_A, size_t nr_cols_A)
{
	for (size_t i = 0; i < nr_rows_A; ++i)
		for (size_t j = 0; j < nr_cols_A; ++j)
			A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}

void fill_cos(float *A, size_t nr_rows_A, size_t nr_cols_A)
{
	for (size_t i = 0; i < nr_rows_A; ++i)
		for (size_t j = 0; j < nr_cols_A; ++j)
			A[i + j * nr_rows_A] = cos(float(i + j * nr_rows_A));
}

#ifdef ROCBLAS_API_H_
// rocBLAS API errors
static const char *_hipGetErrorEnum(hipblasStatus_t error)
{
	switch (error)
	{
	case HIPBLAS_STATUS_SUCCESS:
		return "HIPBLAS_STATUS_SUCCESS";

	case HIPBLAS_STATUS_NOT_INITIALIZED:
		return "HIPBLAS_STATUS_NOT_INITIALIZED";

	case HIPBLAS_STATUS_ALLOC_FAILED:
		return "HIPBLAS_STATUS_ALLOC_FAILED";

	case HIPBLAS_STATUS_INVALID_VALUE:
		return "HIPBLAS_STATUS_INVALID_VALUE";

	case HIPBLAS_STATUS_ARCH_MISMATCH:
		return "HIPBLAS_STATUS_ARCH_MISMATCH";

	case HIPBLAS_STATUS_MAPPING_ERROR:
		return "HIPBLAS_STATUS_MAPPING_ERROR";

	case HIPBLAS_STATUS_EXECUTION_FAILED:
		return "HIPBLAS_STATUS_EXECUTION_FAILED";

	case HIPBLAS_STATUS_INTERNAL_ERROR:
		return "HIPBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}
#endif

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(rocblas_handle handle, const float *A, const float *B, float *C, const int m, const int k, const int n)
{
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	// hipblasStatus_t err = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	rocblas_status stat = rocblas_sgemm(
		handle,
		rocblas_operation_none,
		rocblas_operation_none,
		m, n, k,
		alpha,
		A, lda,
		B, ldb,
		beta,
		C, ldc);

	// if (err != HIPBLAS_STATUS_SUCCESS)
	//	std::cout << "Error: " <<  _hipGetErrorEnum(err) << std::endl;

	if (stat != rocblas_status_success)
	{
		std::cout << "ERROR: sgemm failed\n";
	}
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A)
{

	for (int i = 0; i < nr_rows_A; ++i)
	{
		for (int j = 0; j < nr_cols_A; ++j)
		{
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int def(int value, int reps, int device)
{

	hipSetDevice(device);
	hipStream_t computeStream;
	hipError_t result;
	result = hipStreamCreate(&computeStream);

	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = value;

	// float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	// float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));

	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	hipMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	hipMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	hipMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	// hipMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),hipMemcpyHostToDevice);
	// hipMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),hipMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	// GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	// GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
	// fill_sin(h_A, nr_rows_A, nr_cols_A);
	// fill_cos(h_B, nr_rows_B, nr_cols_B);

	// Adding ability to read the data generated directly from disk to reduce
	// gpu utilizaion between multiple runs
	std::stringstream ss;
	ss << value;
	std::string fname_A = std::string("host_A_") + ss.str() + std::string(".bin");
	std::string fname_B = std::string("host_B_") + ss.str() + std::string(".bin");

	float *h_A;
	float *h_B;
	h_A = read_from_file(fname_A);
	h_B = read_from_file(fname_B);

	// Timer to check how long memory copies take
	GpuTimer memcpy_timer;
	memcpy_timer.Start();

	// Optionally we can copy the data back on CPU and print the arrays
	hipMemcpyAsync(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), hipMemcpyHostToDevice, computeStream);
	hipMemcpyAsync(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), hipMemcpyHostToDevice, computeStream);
	// std::cout << "A =" << std::endl;
	// print_matrix(h_A, nr_rows_A, nr_cols_A);
	// std::cout << "B =" << std::endl;
	// print_matrix(h_B, nr_rows_B, nr_cols_B);

	// Ensure memcpy completes before kernel launches by synchronizing
	hipStreamSynchronize(computeStream);
	memcpy_timer.Stop();
	// std::cout << "Memcpy Time:" << memcpy_timer.Elapsed() << std::endl;

	// Create a handle for CUBLAS
	rocblas_handle handle;
	rocblas_create_handle(&handle);
	rocblas_set_stream(handle, computeStream);
	GpuTimer timer;

	// Warm Up Kernel
	gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	hipStreamSynchronize(computeStream);

	float counter = 0;
	std::time_t curr_time = std::time(nullptr);
	std::cout << "Kernel 0 Start Time: " << curr_time << std::endl;

	for (int i = 0; i < reps; i++)
	{
		// Multiply A and B on GPU
		timer.Start();
		gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
		hipStreamSynchronize(computeStream);
		timer.Stop();
		// if (i >= 5)
		// 	counter += timer.Elapsed();
		// std::cout << "Kernel " << i << " Runtime = " << timer.Elapsed() << std::endl;
	}

	// Print out average kernel duration for the 100 kernels
	// std::cout << (float)counter / (reps - 5) << std::endl;

	// Destroy the handle
	rocblas_destroy_handle(handle);

	// Copy (and print) the result on host memory
	hipMemcpyAsync(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), hipMemcpyDeviceToHost, computeStream);
	// std::cout << "C =" << std::endl;
	// print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);

	result = hipStreamDestroy(computeStream);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int main(int argc, char *argv[])
{
	// for (int i=100; i <= 100000; i = i*10){
	// 	std::cout << "\n\n\n" << i << "\n";
	// 	def(1024, i);
	// }
	if (argc != 4)
	{
		std::cout << "Usage: mul <dim> <reps> <target-device num>" << std::endl;
		exit(-1);
	}
	int dim = atoi(argv[1]);
	int reps = atoi(argv[2]);
	int device = atoi(argv[3]);
	//cout << dim <<
	def(dim, reps, device);
	return 0;
}
