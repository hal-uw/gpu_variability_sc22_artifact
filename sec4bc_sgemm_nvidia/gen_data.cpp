#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <cmath>

using namespace std;

// Randomization helpers
// adapted from https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/rocm-3.0/clients/include/rocblas_init.hpp#L42

void fill_sin(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}


void fill_cos(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = cos(float(i + j * nr_rows_A));
}

void verify(float *A, float *B, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	        if (A[i + j * nr_rows_A] != B[i + j * nr_rows_A]){
            cout << "Verification ERROR: " << i << " " << j << " " << nr_rows_A;
            cout << endl;
            return;
          }
}


// Function to write binary data to file
void write_to_file(string file_name, float* data, size_t size){
  // lets write it to binary file
  ofstream ofile;

  // open a binary file
  ofile.open(file_name, ios::binary | ios::out);

  //write the data to file
  ofile.write((char*) data, size);

  // close the file
  ofile.close();

  return;
}

// Function to read data from file
float* read_from_file(string file_name){
  // lets get filesize
  struct stat results;
  if (stat(file_name.c_str(), &results) != 0){
    // An error occurred
    cout << "ERROR: unable to get filesize" << endl;
    return NULL;
  }
  // The size of the file in bytes is in results.st_size
  // Lets allocate an array to contain the binary file
  float* data = (float *)malloc(results.st_size);


  // lets write it to binary file
  ifstream infile;

  // open a binary file
  infile.open(file_name, ios::binary | ios::in);

  //read data from file
  infile.read((char*) data, results.st_size);

  // close the file
  infile.close();

  return data;
}



int main(int argc, char* argv[])
{
  if (argc != 2){
		std::cout << "Usage: gen_data <dim>" << std::endl;
		exit(-1);
	}

  int dim = atoi(argv[1]);
  cout << dim << endl;

  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
  // float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
  //
  cout << "SIZE: " << nr_rows_A <<" " <<  sizeof(float) << endl;

  fill_sin(h_A, nr_rows_A, nr_cols_A);
	fill_cos(h_B, nr_rows_B, nr_cols_B);

  string fname_A = string("host_A_") + string(argv[1]) +string(".bin");
  string fname_B = string("host_B_") + string(argv[1]) +string(".bin");

  write_to_file(fname_A, h_A, nr_rows_A * nr_cols_A * sizeof(float));
  write_to_file(fname_B, h_B, nr_rows_B * nr_cols_B * sizeof(float));

  float* test = read_from_file(fname_A);
  float* testB = read_from_file(fname_B);

  // verification
  verify(test, h_A, nr_rows_A, nr_cols_A);
  verify(testB, h_B, nr_rows_B, nr_cols_B);

  return 0;

}
