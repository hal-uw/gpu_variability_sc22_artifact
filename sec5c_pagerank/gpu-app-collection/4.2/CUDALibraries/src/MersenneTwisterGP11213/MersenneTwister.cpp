/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.  
 */

// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA Error handling

#include <shrQATest.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime_api.h>
#include <curand.h>

float compareResults(int rand_n, float *h_RandGPU, float *h_RandCPU);

const int    DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCurandErrors(err)           __checkCurandErrors (err, __FILE__, __LINE__)

    inline void __checkCurandErrors( curandStatus_t err, const char *file, const int line )
    {
        if( CURAND_STATUS_SUCCESS != err) {
            fprintf(stderr, "%s(%i) : checkCurandErrors() CURAND error %d: ", file, line, (int)err);
            switch (err) {
                case CURAND_STATUS_VERSION_MISMATCH:    fprintf(stderr, "CURAND_STATUS_VERSION_MISMATCH");
                case CURAND_STATUS_NOT_INITIALIZED:     fprintf(stderr, "CURAND_STATUS_NOT_INITIALIZED");
                case CURAND_STATUS_ALLOCATION_FAILED:   fprintf(stderr, "CURAND_STATUS_ALLOCATION_FAILED");
                case CURAND_STATUS_TYPE_ERROR:          fprintf(stderr, "CURAND_STATUS_TYPE_ERROR");
                case CURAND_STATUS_OUT_OF_RANGE:        fprintf(stderr, "CURAND_STATUS_OUT_OF_RANGE"); 
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE: fprintf(stderr, "CURAND_STATUS_LENGTH_NOT_MULTIPLE");
                case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: 
				                fprintf(stderr, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED");
                case CURAND_STATUS_LAUNCH_FAILURE:      fprintf(stderr, "CURAND_STATUS_LAUNCH_FAILURE"); 
                case CURAND_STATUS_PREEXISTING_FAILURE: fprintf(stderr, "CURAND_STATUS_PREEXISTING_FAILURE");
                case CURAND_STATUS_INITIALIZATION_FAILED:     
				                fprintf(stderr, "CURAND_STATUS_INITIALIZATION_FAILED");
                case CURAND_STATUS_ARCH_MISMATCH:       fprintf(stderr, "CURAND_STATUS_ARCH_MISMATCH");
                case CURAND_STATUS_INTERNAL_ERROR:      fprintf(stderr, "CURAND_STATUS_INTERNAL_ERROR");
                default: fprintf(stderr, "CURAND Unknown error code\n");
            }
            exit(-1);
        }
    }
// end of CUDA Helper Functions

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    shrQAStart(argc, argv);

    // initialize the GPU, either identified by --device
    // or by picking the device with highest flop rate.
    int devID = findCudaDevice(argc, (const char **)argv);

    // parsing the number of random numbers to generate
    int rand_n = DEFAULT_RAND_N;
    if( checkCmdLineFlag(argc, (const char**) argv, "count") )  
    {       
        rand_n = getCmdLineArgumentInt(argc, (const char**) argv, "count"); 
    }
    printf("Allocating data for %i samples...\n", rand_n);
     
    // parsing the seed
    int seed = DEFAULT_SEED;
    if( checkCmdLineFlag(argc, (const char**) argv, "seed") ) 
    {       
        seed = getCmdLineArgumentInt(argc, (const char**) argv, "seed"); 
    }
    printf("Seeding with %i ...\n", seed);
    

    float *d_Rand; 
    checkCudaErrors( cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)) );
    
    curandGenerator_t prngGPU;
    checkCurandErrors( curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32) ); 
    checkCurandErrors( curandSetPseudoRandomGeneratorSeed(prngGPU, seed) );

    curandGenerator_t prngCPU;
    checkCurandErrors( curandCreateGeneratorHost(&prngCPU, CURAND_RNG_PSEUDO_MTGP32) ); 
    checkCurandErrors( curandSetPseudoRandomGeneratorSeed(prngCPU, seed) );

    //
    // Example 1: Compare random numbers generated on GPU and CPU
    float *h_RandGPU  = (float *)malloc(rand_n * sizeof(float));

    printf("Generating random numbers on GPU...\n\n");
    checkCurandErrors( curandGenerateUniform(prngGPU, (float*) d_Rand, rand_n) );

    printf("\nReading back the results...\n");
    checkCudaErrors( cudaMemcpy(h_RandGPU, d_Rand, rand_n * sizeof(float), cudaMemcpyDeviceToHost) );

    
    float *h_RandCPU  = (float *)malloc(rand_n * sizeof(float));
     
    printf("Generating random numbers on CPU...\n\n");
    checkCurandErrors( curandGenerateUniform(prngCPU, (float*) h_RandCPU, rand_n) ); 
 
    printf("Comparing CPU/GPU random numbers...\n\n");
    float L1norm = compareResults(rand_n, h_RandGPU, h_RandCPU); 
    
    //
    // Example 2: Timing of random number generation on GPU
    const int numIterations = 10;
    int i;
    StopWatchInterface *hTimer;

    checkCudaErrors( cudaDeviceSynchronize() );
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < numIterations; i++)
    {
        checkCurandErrors( curandGenerateUniform(prngGPU, (float*) d_Rand, rand_n) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    sdkStopTimer(&hTimer);

    double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer)/(double)numIterations;

    printf("MersenneTwister, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers\n", 
               1.0e-9 * rand_n / gpuTime, gpuTime, rand_n); 

    printf("Shutting down...\n");

    checkCurandErrors( curandDestroyGenerator(prngGPU) );
    checkCurandErrors( curandDestroyGenerator(prngCPU) );
    checkCudaErrors( cudaFree(d_Rand) );
    sdkDeleteTimer( &hTimer);
    free(h_RandGPU);
    free(h_RandCPU);

    cudaDeviceReset();	
    shrQAFinishExit(argc, (const char**)argv, (L1norm < 1e-6) ? QA_PASSED : QA_FAILED);
}


float compareResults(int rand_n, float* h_RandGPU, float* h_RandCPU)
{
    int i;
    float rCPU, rGPU, delta;
    float max_delta = 0.;
    float sum_delta = 0.;
    float sum_ref   = 0.;
    for(i = 0; i < rand_n; i++)
    {
        rCPU = h_RandCPU[i];
        rGPU = h_RandGPU[i];
        delta = fabs(rCPU - rGPU);
        sum_delta += delta;
        sum_ref   += fabs(rCPU);
        if(delta >= max_delta) max_delta = delta;
    }
    float L1norm = (float)(sum_delta / sum_ref);
    printf("Max absolute error: %E\n", max_delta);
    printf("L1 norm: %E\n\n", L1norm);

    return L1norm;
}
