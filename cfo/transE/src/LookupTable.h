#ifndef LOOKUPTABLE_H
#define LOOKUPTABLE_H

#include <fstream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>

__global__ void random(float *m, float param, int N) {
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init(clock64(), threadIdx.x, 0, &state);
  if (blockId < N) {
    *(m + threadId) = (curand(&state) / (float)UINT_MAX - 0.5) * 2 * param;
  }
}

__global__ void L2Norm(float *m, float *l2, int rows, int cols) {
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  float val = 0;
  if (blockId < rows) {
    for (int i = 0; i < cols; i++) {
      val += (*(m + blockId * cols + i)) * (*(m + blockId * cols + i));
    }
    *(l2 + blockId) = sqrtf(val);
  }
}

__global__ void normalize(float *m, float *l2, int rows, int cols) {
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = threadIdx.x;
  if (blockId < rows && threadId < cols) {
    *(m + blockId * cols + threadId) /= *(l2 + blockId);
  }
}

__global__ void sum(float *m, float *l2, int rows, int cols) {
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  float val = 0;
  if (blockId < rows) {
    for (int i = 0; i < cols; i++) {
      val += (*(m + blockId * cols + i)) * (*(m + blockId * cols + i));
    }
    *(l2 + blockId) = val;
  }
}

struct LookupTable {
  int rows;
  int cols;
  float *d_table;

  LookupTable(int r, int c);
  ~LookupTable();
  void save(std::string &path);
};

LookupTable::LookupTable(int r, int c) {
  rows = r;
  cols = c;
  cudaMalloc((void**)&d_table, sizeof(float) * cols * rows);
  float param = 6 / pow(cols, 0.5);
  dim3 blocksPerGrim(2000, 2000);
  random<<<blocksPerGrim, cols>>>(d_table, param, rows);

  float *l2_norm;
  cudaMalloc((void**)&l2_norm, sizeof(float) * rows);
  L2Norm<<<blocksPerGrim, 1>>>(d_table, l2_norm, rows, cols);
  normalize<<<blocksPerGrim, cols>>>(d_table, l2_norm, rows, cols);
/*
  sum<<<blocksPerGrim, 1>>>(d_table, l2_norm, rows, cols);
  float *h_l2 = (float*)malloc(sizeof(float) * rows);
  cudaMemcpy(h_l2, l2_norm, sizeof(float) * rows, cudaMemcpyDeviceToHost);
  cudaFree(l2_norm);
  printf("LookupTable: %s\n", cudaGetErrorString(cudaGetLastError()));
  for (int i = 0; i < rows; i++) {
    std::cout << h_l2[i] << std::endl;
  }
*/
}

LookupTable::~LookupTable() {
  cudaFree(d_table);
}

void LookupTable::save(std::string &path) {
  float *h_table = (float*)malloc(sizeof(float) * rows * cols);
  cudaMemcpy(h_table, d_table, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

  std::ofstream fout(path, std::ofstream::out | std::ofstream::binary);
  fout.write((char*)h_table, sizeof(float) * rows * cols);
  fout.close();
  free(h_table);
}

#endif

