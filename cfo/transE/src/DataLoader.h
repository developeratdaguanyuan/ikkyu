#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "BatchDataLoader.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class DataLoader {
private:
  int size;
  int batchSize;

  vector<int> sub;
  vector<int> prd;
  vector<int> obj;

  int idx = 0;
  vector<int> rank;

private:
  void randperm(vector<int> &perm);

public:
  int maxEnt, maxPrd;

public:
  int loadData(string path);
  void setBatchSize(int s);
  int getBatchSize() {
    return batchSize;
  }
  int getSize() {
    return size;
  }
  void nextBatch(BatchDataLoader &batch_data_loader);
};

void DataLoader::setBatchSize(int s) {
  batchSize = s;
}

int DataLoader::loadData(string path) {
  std::ifstream reader(path.c_str(), std::ifstream::in);
  int s, p, o;
  maxEnt = -1, maxPrd = -1;
  while (reader >> s >> p >> o) {
    sub.push_back(s);
    prd.push_back(p);
    obj.push_back(o);
    maxEnt = maxEnt > (s > o ? s : o) ? maxEnt : (s > o ? s : o);
    maxPrd = maxPrd > p ? maxPrd : p;
  }
  reader.close();
  size = sub.size();

  cout << "triple size: " << size << endl;
  cout << "max Entity ID: " << maxEnt << endl;
  cout << "max Predicate ID: " << maxPrd << endl;
  
  return 0;
}

void DataLoader::randperm(vector<int> &perm) {
  if (perm.size() == 0) {
    for(int i = 0; i < size; i++) {
      perm.push_back(i);
    }
  }
  time_t t;
  srand((unsigned)time(&t));
  for(int i = 0; i < size; i++) {
    int j = rand() % (size - i) + i;
    int t = perm[j];
    perm[j] = perm[i];
    perm[i] = t;
  }
}

void DataLoader::nextBatch(BatchDataLoader &batch_data_loader) {
  if (rank.size() == 0 || idx + batchSize > size) {
    idx = 0;
    randperm(rank);
  }
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < batch_data_loader.rows; i++) {
    int index = rank.at(idx + i);
    batch_data_loader.sub[i] = sub.at(index);
    batch_data_loader.prd[i] = prd.at(index);
    batch_data_loader.obj[i] = obj.at(index);
    int id_t;
    id_t = rand() % maxEnt + 1;
    batch_data_loader.nsub[i] =
      (id_t != batch_data_loader.sub[i]) ? id_t : rand() % maxEnt + 1;
    id_t = rand() % maxPrd + 1;
    batch_data_loader.nprd[i] =
      (id_t != batch_data_loader.prd[i]) ? id_t : rand() % maxPrd + 1;
    id_t = rand() % maxEnt + 1;
    batch_data_loader.nobj[i] =
      (id_t != batch_data_loader.obj[i]) ? id_t : rand() % maxEnt + 1;
  }

  idx += batchSize;
}


#endif


