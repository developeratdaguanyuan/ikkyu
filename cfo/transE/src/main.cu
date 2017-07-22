#include "DataLoader.h"
#include "BatchDataLoader.h"
#include "LookupTable.h"
#include "transE.h"

#include <iostream>

#define BATCHSIZE 100
#define DIMENSION 256
#define VALIDSIZE 50000
#define LEARNRATE 0.01

int main() {
  string model_dir = "model/";
  string train_data = "data/FB5M-triples.txt";

  // Load Train Data
  DataLoader train_data_loader;
  train_data_loader.setBatchSize(BATCHSIZE);
  train_data_loader.loadData(train_data);
  
  // Load Valid Data
  ValidDataLoader valid_data_loader(VALIDSIZE);

  // Batch Data Loader
  BatchDataLoader batch_data_loader(BATCHSIZE, DIMENSION);

  // Build Entities&Relations LookupTable
  LookupTable entities(train_data_loader.maxEnt + 1, DIMENSION);
  LookupTable relations(train_data_loader.maxPrd + 1, DIMENSION);

  TransE transE(&train_data_loader, &valid_data_loader, &batch_data_loader,
                &entities, &relations, 1000, 10, LEARNRATE, model_dir);
  transE.train();
}

