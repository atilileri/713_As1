/*
* 713_Assignment 1
* In the assignment, you will implement a matrix multiplication algorithm with CUDA.
*
* Code from the CUDA C Programming Guide is studied but, not copied.
* Algortihm and matrix multiplication strategy is my own.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


// Thread block size
#define M 5 /* Height of A */
#define N 9 /* Width  of A and Height of B */
#define P 3 /* Width  of B */

// Matrix multiplication kernel device function called by MatrixMult()
__global__ void kernelMatrixMult(int *A, int *B, int *C)
{
	// Each thread calculates C[row][col]
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int temp = 0;
	// Return if size is reached
	if (row >= M || col >= P) return;
	//multiply every element and add to a temporary variable
	for (int i = 0; i < N; i++)
	{
		temp += A[(row * N) + i] * B[col + (i * P)];
	}
	C[(row * P) + col] = temp;
}

// Matrix multiplication host function code
void MatrixMult(const int A[M * N], const int B[N * P], int C[M * P])
{
	int *d_A;
	int *d_B;
	int *d_C;
	
	cudaError_t err;
	size_t size;
	
	//Allocate A in device memory
	size = M * N * sizeof(int);
	err = cudaMalloc(&d_A, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	//Copy A from host memory
	err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	//Allocate B in device memory
	size = N * P * sizeof(int);
	err = cudaMalloc(&d_B, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	//Copy B from host memory
	err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));

	//Allocate C in device memory
	size = M * P * sizeof(int);
	err = cudaMalloc(&d_C, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
	//Copy C from host memory
	err = cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
	printf("Copy C to device: %s\n", cudaGetErrorString(err));

	// Invoke kernel
	dim3 dimBlock(1);
	dim3 dimGrid(P, M);
	kernelMatrixMult << <dimGrid, dimBlock >> >(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Kernel Sync: %s\n", cudaGetErrorString(err));

	// Read C from device memory
	err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(err));
	
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main(int argc, char* argv[])
{
	int h_A[M * N];
	int h_B[N * P];
	int h_C[M * P];
	
	// Fill matrices
	for (int i = 0; i < M * N; i++)
		h_A[i] = rand() % 10;

	for (int i = 0; i < N * P; i++)
		h_B[i] = rand() % 20;

	MatrixMult(h_A, h_B, h_C);

	// Print matrices
	for (int i = 0; i < M * N; i++)
	{
		if (!(i%N)) printf("\n");
		printf("%d ", h_A[i]);
	}
	printf("\n");

	for (int i = 0; i < N * P; i++)
	{
		if (!(i%P)) printf("\n");
		printf("%d ", h_B[i]);
	}
	printf("\n");

	for (int i = 0; i < M * P; i++)
	{
		if (!(i%P)) printf("\n");
		printf("%d ", h_C[i]);
	}
	printf("\n");

	//wait for enter
	getchar();
}

