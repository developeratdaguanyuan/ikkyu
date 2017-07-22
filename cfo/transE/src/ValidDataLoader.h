#ifndef VALID_DATA_LOADER_H
#define VALID_DATA_LOADER_H

struct ValidDataLoader {
  int size;
  int *sub, *prd, *obj;
  int *devSub, *devPrd, *devObj;
  
  ValidDataLoader(int);
  ~ValidDataLoader();
  int loadData(string);
};

ValidDataLoader::ValidDataLoader(int sz) {
  size = sz;
  sub = (int *)malloc(sizeof(int) * size);
  prd = (int *)malloc(sizeof(int) * size);
  obj = (int *)malloc(sizeof(int) * size);
  cudaMalloc((void**)&(devSub), sizeof(int) * size);
  cudaMalloc((void**)&(devPrd), sizeof(int) * size);
  cudaMalloc((void**)&(devObj), sizeof(int) * size);
}

ValidDataLoader::~ValidDataLoader() {
  free(sub);
  free(prd);
  free(obj);
  cudaFree(devSub);
  cudaFree(devPrd);
  cudaFree(devObj);
}

int ValidDataLoader::loadData(string path) {
  std::ifstream reader(path.c_str(), std::ifstream::in);
  int i = 0;
  int s, p, o;
  while (reader >> s >> p >> o) {
    if (i >= size) {
      std::cout << "Valid data is too large!!!";
      break;
    }
    sub[i] = s;
    prd[i] = p;
    obj[i] = o;
    ++i;
  }
  reader.close();

  cudaMemcpy(devSub, sub, sizeof(int) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(devPrd, prd, sizeof(int) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(devObj, obj, sizeof(int) * size, cudaMemcpyHostToDevice);

  return 0;
}

#endif
