#ifndef BATCH_DATA_LOADER_H
#define BATCH_DATA_LOADER_H

#include "LookupTable.h"

__global__ void getEmbedding(float *des, int *idx, float *src) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  *(des + i * blockDim.x + j) = *(src + idx[i] * blockDim.x + j);
}

struct BatchDataLoader {
  int rows, cols;
  int *sub, *prd, *obj;
  int *nsub, *nprd, *nobj;
  int *devSub, *devPrd, *devObj;
  int *devNSub, *devNPrd, *devNObj;
  float *devSubEmbed, *devPrdEmbed, *devObjEmbed;
  float *devNSubEmbed, *devNPrdEmbed, *devNObjEmbed;

  BatchDataLoader(int, int);
  ~BatchDataLoader();
  void lookuptable(LookupTable*, LookupTable*);
};

BatchDataLoader::BatchDataLoader(int r, int c) {
  rows = r;
  cols = c;

  int index_size = sizeof(int) * rows;
  sub = (int*)malloc(index_size);
  prd = (int*)malloc(index_size);
  obj = (int*)malloc(index_size);
  nsub = (int*)malloc(index_size);
  nprd = (int*)malloc(index_size);
  nobj = (int*)malloc(index_size);

  cudaMalloc((void**)&devSub, index_size);
  cudaMalloc((void**)&devPrd, index_size);
  cudaMalloc((void**)&devObj, index_size);
  cudaMalloc((void**)&devNSub, index_size);
  cudaMalloc((void**)&devNPrd, index_size);
  cudaMalloc((void**)&devNObj, index_size);

  int embed_size = sizeof(float) * rows * cols;
  cudaMalloc((void**)&devSubEmbed, embed_size);
  cudaMalloc((void**)&devPrdEmbed, embed_size);
  cudaMalloc((void**)&devObjEmbed, embed_size);
  cudaMalloc((void**)&devNSubEmbed, embed_size);
  cudaMalloc((void**)&devNPrdEmbed, embed_size);
  cudaMalloc((void**)&devNObjEmbed, embed_size);
}

BatchDataLoader::~BatchDataLoader() {
  free(sub);
  free(prd);
  free(obj);
  free(nsub);
  free(nprd);
  free(nobj);
  cudaFree(devSub);
  cudaFree(devPrd);
  cudaFree(devObj);
  cudaFree(devNSub);
  cudaFree(devNPrd);
  cudaFree(devNObj);
  cudaFree(devSubEmbed);
  cudaFree(devPrdEmbed);
  cudaFree(devObjEmbed);
  cudaFree(devNSubEmbed);
  cudaFree(devNPrdEmbed);
  cudaFree(devNObjEmbed);
}

void BatchDataLoader::lookuptable(
  LookupTable *p_entities, LookupTable *p_relations) {

  int ss = sizeof(int) * rows;
  cudaMemcpy(devSub, sub, ss, cudaMemcpyHostToDevice);
  cudaMemcpy(devPrd, prd, ss, cudaMemcpyHostToDevice);
  cudaMemcpy(devObj, obj, ss, cudaMemcpyHostToDevice);
  cudaMemcpy(devNSub, nsub, ss, cudaMemcpyHostToDevice);
  cudaMemcpy(devNPrd, nprd, ss, cudaMemcpyHostToDevice);
  cudaMemcpy(devNObj, nobj, ss, cudaMemcpyHostToDevice);

  getEmbedding<<<rows, cols>>>(devSubEmbed, devSub, p_entities->d_table);
  getEmbedding<<<rows, cols>>>(devPrdEmbed, devPrd, p_relations->d_table);
  getEmbedding<<<rows, cols>>>(devObjEmbed, devObj, p_entities->d_table);
  getEmbedding<<<rows, cols>>>(devNSubEmbed, devNSub, p_entities->d_table);
  getEmbedding<<<rows, cols>>>(devNPrdEmbed, devNPrd, p_relations->d_table);
  getEmbedding<<<rows, cols>>>(devNObjEmbed, devNObj, p_entities->d_table);
}

#endif

