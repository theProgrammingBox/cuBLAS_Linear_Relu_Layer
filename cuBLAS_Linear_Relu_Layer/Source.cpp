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

void MatMulMat(cublasHandle_t handle, size_t a, size_t b, size_t c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &ONE, B, c, A, b, &ZERO, C, c);	// doing row-major math using column major apis
}

void MatTMulMat(cublasHandle_t handle, size_t a, size_t b, size_t c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, c, a, b, &ONE, B, c, A, a, &ZERO, C, c);	// doing row-major math using column major apis
}

void MatMulMatT(cublasHandle_t handle, size_t a, size_t b, size_t c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, c, a, b, &ONE, B, b, A, b, &ZERO, C, c);	// doing row-major math using column major apis
}

void RandFillMat(curandGenerator_t randomGenerator, float* matrix, size_t size, float mean = 0.0f, float deviation = 1.0f) {
	curandGenerateNormal(randomGenerator, matrix, size + (size & 1), mean, deviation);	// required to be even, may cause a bug
}

__global__ void CudaRelu(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t index = 1024 * blockIdx.x + threadIdx.x;
	if (index < size) {
		float item = inputMatrix[index];
		outputMatrix[index] = (item > 0) * item;
	}
}

void Relu(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t blocks = 0.0009765625f * size + 1;	// I think this can't exceed 2147483647
	dim3 threads(1024);							// x * y * z can't exceed 1024 it seems
	CudaRelu <<<blocks, threads>>> (inputMatrix, outputMatrix, size);
}

__global__ void CudaReluDerivative(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t index = 1024 * blockIdx.x + threadIdx.x;
	if (index < size) {
		float item = inputMatrix[index];
		outputMatrix[index] = (item > 0);
	}
}

void ReluDerivative(float* inputMatrix, float* outputMatrix, size_t size) {
	size_t blocks = 0.0009765625f * size + 1;	// I think this can't exceed 2147483647
	dim3 threads(1024);							// x * y * z can't exceed 1024 it seems
	CudaReluDerivative <<<blocks, threads>>> (inputMatrix, outputMatrix, size);
}

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