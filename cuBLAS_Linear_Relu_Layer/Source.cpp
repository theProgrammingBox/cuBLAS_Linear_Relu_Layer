#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

const float ONE = 1.0f;
const float ZERO = 0.0f;

// In c++, all matrices are stored in row-major order
// These functions do run column-major operations with row-major matrices
// If the function transposes the matrix, pass in the original matrix and the function will "transpose" it for you

void MatMulMat(cublasHandle_t handle, size_t matrix1Rows, size_t matrix1Columns, size_t matrix2Columns, float* matrix1, float* matrix2, float* outputMatrix) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix2Columns, matrix1Rows, matrix1Columns, &ONE, matrix2, matrix2Columns, matrix1, matrix1Columns, &ZERO, outputMatrix, matrix2Columns);	// doing row-major math using column major apis
	// A * B = C					// A is matrix1, B is matrix2, C is outputMatrix
	// (a x b) * (b x c) = (a x c)	// operation equation
	// (a x b) * (b x c) = (a x c)	// original matrix dimensions
	// input is a, b, c, A, B, C
}

void MatTMulMat(cublasHandle_t handle, size_t matrix1Columns, size_t matrix1Rows, size_t matrix2Columns, float* matrix1, float* matrix2, float* outputMatrix) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, matrix2Columns, matrix1Columns, matrix1Rows, &ONE, matrix2, matrix2Columns, matrix1, matrix1Columns, &ZERO, outputMatrix, matrix2Columns);	// doing row-major math using column major apis
	// A ^ T * B = C				// A is matrix1, B is matrix2, C is outputMatrix
	// (b x a) * (a x c) = (b x c)	// operation equation
	// (a x b) * (a x c) = (b x c)	// original matrix dimensions
	// input is b, a, c, A, B, C
}

void MatMulMatT(cublasHandle_t handle, size_t matrix1Rows, size_t matrix1Columns, size_t matrix2Rows, float* matrix1, float* matrix2, float* outputMatrix) {
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix2Rows, matrix1Rows, matrix1Columns, &ONE, matrix2, matrix1Columns, matrix1, matrix1Columns, &ZERO, outputMatrix, matrix2Rows);	// doing row-major math using column major apis
	// A * B ^ T = C				// A is matrix1, B is matrix2, C is outputMatrix
	// (a x b) * (b x c) = (a x c)	// operation equation
	// (a x b) * (c x b) = (a x c)	// original matrix dimensions
	// input is a, b, c, A, B, C
}

void RandFillMat(curandGenerator_t randomGenerator, float* matrix, size_t size, float mean = 0.0f, float deviation = 1.0f) {
	curandGenerateNormal(randomGenerator, matrix, size + (size & 1), mean, deviation);	// required to be even, may cause matrix1Rows bug
}

__global__ void CudaLinearClipTahn(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t index = 1024 * blockIdx.x + threadIdx.x;
	if (index < size) {
		float input = inputMatrix[index] + 1;
		input = (input > 0) * input - 2;
		outputMatrix[index] = (input < 0) * input + 1;
		// -1 if input < -1, 1 if input > 1, otherwise input
		// outputMatrix[index] = input < -1 ? -1 : (input > 1 ? 1 : input);
	}
}

void LinearClipTahn(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t blocks = 0.0009765625f * size + 1;	// I think this can't exceed 2147483647
	dim3 threads(1024);							// x * y * z can't exceed 1024 it seems
	CudaLinearClipTahn <<<blocks, threads>>> (inputMatrix, outputMatrix, size);
}

__global__ void CudaLinearClipTahnDerivative(float* inputMatrix, float* gradientMatrix, float* outputMatrix, size_t size) {
	size_t index = 1024 * blockIdx.x + threadIdx.x;
	if (index < size) {
		float input = inputMatrix[index];
		float gradient = gradientMatrix[index];
		bool go = input < 1;	// greater than 1
		outputMatrix[index] = (1 - (go ^ (input > -1)) * (go ^ (gradient > 0))) * gradient;
		// basically if input is greater than 1 and gradient is greater than 0, or input is less than -1 and gradient is less than 0, then output is 0
		// just two sided relu that allows gradient to propagate if it brings the input closer to the boundary
		// outputMatrix[index] = (input > 1 && gradient > 0) || (input < -1 && gradient < 0) ? 0 : gradient;
	}
}

void LinearClipTahnDerivative(float* inputMatrix, float* gradientMatrix, float* outputMatrix, size_t size) {
	size_t blocks = 0.0009765625f * size + 1;	// I think this can't exceed 2147483647
	dim3 threads(1024);							// x * y * z can't exceed 1024 it seems
	CudaLinearClipTahnDerivative <<<blocks, thread>>> (inputMatrix, gradientMatrix, outputMatrix, size);
}

// A linear layer followed by a clip tahn activation function is as follows:
// MatMulMat(handle, batchSize, inputFeatures, outputFeatures, gpuInputMatrix, gpuWeightMatrix, gpuOutputMatrix);
// LinearClipTahn(gpuOutputMatrix, gpuActivatedOutputMatrix, batchSize * outputFeatures);

// The derivative of the linear layer followed by a clip tahn activation function is as follows:
// LinearClipTahnDerivative(gpuActivatedOutputMatrix, gpuActivatedOutputGradientMatrix, gpuOutputGradientMatrix, batchSize * outputFeatures);
// MatTMulMat(handle, inputFeatures, batchSize, outputFeatures, gpuInputMatrix, gpuOutputGradientMatrix, gpuWeightGradientMatrix);
// MatMulMatT(handle, batchSize, outputFeatures, inputFeatures, gpuOutputGradientMatrix, gpuWeightMatrix, gpuInputGradientMatrix);

// To update the weights, use the following from cublas:
// cublasSaxpy(handle, inputFeatures * outputFeatures, &learningRate, gpuWeightGradientMatrix, 1, gpuWeightMatrix, 1);

int main() {
	curandGenerator_t randomGenerator;
	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	cublasHandle_t handle;
	cublasCreate(&handle);

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const size_t numIterations = 100;

	size_t batchSize = 1 << 8;
	size_t inputFeatures = 1 << 10;
	size_t outputFeatures = 1 << 9;

	float* gpuInputMatrix, * gpuWeightMatrix, * gpuOutputMatrix;
	cudaMallocManaged(&gpuInputMatrix, batchSize * inputFeatures * sizeof(float));
	cudaMallocManaged(&gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float));
	cudaMallocManaged(&gpuOutputMatrix, batchSize * outputFeatures * sizeof(float));

	float* cpuInputMatrix = new float[batchSize * inputFeatures];
	float* cpuWeightMatrix = new float[inputFeatures * outputFeatures];
	float* cpuOutputMatrix = new float[batchSize * outputFeatures];

	float* output;



	// matrix times matrix
	RandFillMat(randomGenerator, gpuInputMatrix, batchSize * inputFeatures);
	RandFillMat(randomGenerator, gpuWeightMatrix, inputFeatures * outputFeatures);

	size_t iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		// ab x bc = ac
		// abc, ABC
		MatMulMat(handle, batchSize, inputFeatures, outputFeatures, gpuInputMatrix, gpuWeightMatrix, gpuOutputMatrix);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);

	output = new float[batchSize * outputFeatures];
	for (size_t i = 0; i < batchSize; i++) {
		for (size_t j = 0; j < outputFeatures; j++) {
			float sum = 0;
			for (size_t k = 0; k < inputFeatures; k++) {
				sum += cpuInputMatrix[i * inputFeatures + k] * cpuWeightMatrix[k * outputFeatures + j];
			}
			output[i * outputFeatures + j] = sum;
		}
	}

	float error = 0;
	for (size_t i = 0; i < batchSize * outputFeatures; i++) {
		error += abs(cpuOutputMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (batchSize * outputFeatures) << endl;
	delete[] output;



	// transposed matrix times matrix
	RandFillMat(randomGenerator, gpuOutputMatrix, batchSize * outputFeatures);
	RandFillMat(randomGenerator, gpuInputMatrix, batchSize * inputFeatures);

	iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		// ba x ac = bc
		// bac, ACB
		MatTMulMat(handle, inputFeatures, batchSize, outputFeatures, gpuInputMatrix, gpuOutputMatrix, gpuWeightMatrix);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);

	output = new float[inputFeatures * outputFeatures];
	for (size_t i = 0; i < inputFeatures; i++) {
		for (size_t j = 0; j < outputFeatures; j++) {
			float sum = 0;
			for (size_t k = 0; k < batchSize; k++) {
				sum += cpuInputMatrix[k * inputFeatures + i] * cpuOutputMatrix[k * outputFeatures + j];
			}
			output[i * outputFeatures + j] = sum;
		}
	}

	error = 0;
	for (size_t i = 0; i < inputFeatures * outputFeatures; i++) {
		error += abs(cpuWeightMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (inputFeatures * outputFeatures) << endl;
	delete[] output;



	// matrix times transposed matrix
	RandFillMat(randomGenerator, gpuOutputMatrix, batchSize * outputFeatures);
	RandFillMat(randomGenerator, gpuWeightMatrix, inputFeatures * outputFeatures);

	iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		// ac x cb = ab
		// acb, CBA
		MatMulMatT(handle, batchSize, outputFeatures, inputFeatures, gpuOutputMatrix, gpuWeightMatrix, gpuInputMatrix);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);

	output = new float[batchSize * inputFeatures];
	for (size_t i = 0; i < batchSize; i++) {
		for (size_t j = 0; j < inputFeatures; j++) {
			float sum = 0;
			for (size_t k = 0; k < outputFeatures; k++) {
				sum += cpuOutputMatrix[i * outputFeatures + k] * cpuWeightMatrix[j * outputFeatures + k];
			}
			output[i * inputFeatures + j] = sum;
		}
	}

	error = 0;
	for (size_t i = 0; i < batchSize * inputFeatures; i++) {
		error += abs(cpuInputMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (batchSize * inputFeatures) << endl;
	delete[] output;



	// relu
	RandFillMat(randomGenerator, gpuInputMatrix, batchSize * inputFeatures);

	float* gpuReluOutput;
	cudaMalloc(&gpuReluOutput, batchSize * inputFeatures * sizeof(float));

	float* cpuReluOutput = new float[batchSize * inputFeatures];

	iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		Relu(gpuInputMatrix, gpuReluOutput, batchSize * inputFeatures);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuReluOutput, gpuReluOutput, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);

	output = new float[batchSize * inputFeatures];
	for (size_t i = 0; i < batchSize * inputFeatures; i++) {
		output[i] = cpuInputMatrix[i] * (cpuInputMatrix[i] > 0);
	}

	error = 0;
	for (size_t i = 0; i < batchSize * inputFeatures; i++) {
		error += abs(cpuReluOutput[i] - output[i]);
		if (abs(cpuReluOutput[i] - output[i]) > 0.0001) {
			cout << "Error at " << i << ": " << cpuReluOutput[i] << " " << output[i] << endl;
		}
	}
	cout << "Average error: " << error / (batchSize * inputFeatures) << endl;
	delete[] output;



	cudaFree(gpuInputMatrix);
	cudaFree(gpuWeightMatrix);
	cudaFree(gpuOutputMatrix);
	delete[] cpuInputMatrix;
	delete[] cpuWeightMatrix;
	delete[] cpuOutputMatrix;

	curandDestroyGenerator(randomGenerator);
	cublasDestroy(handle);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}