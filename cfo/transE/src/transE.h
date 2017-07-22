#ifndef TRANSE_H
#define TRANSE_H

#include "DataLoader.h"
#include "LookupTable.h"
#include "ValidDataLoader.h"

#include <set>

// Sub + Prd - Obj
__global__ void LinearForward(float *inputSub, float *inputPrd,
                              float *inputObj, float *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  *(output + i) = *(inputSub + i) + *(inputPrd + i) - *(inputObj + i);
}

__global__ void LinearBackward(float *grandOutput, float *grandInputSub,
                               float *grandInputPrd, float *grandInputObj) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  *(grandInputSub + offset) += *(grandOutput + offset);
  *(grandInputPrd + offset) += *(grandOutput + offset);
  *(grandInputObj + offset) -= *(grandOutput + offset);
}


// L2
__global__ void L2Forward(float *input, float *output, int col) {
  int i = blockIdx.x;
  int s = i * col;
  float val = 0;
  for (int n = 0; n < col; n++) {
    val += (*(input + s + n)) * (*(input + s + n));
  }
  output[i] = sqrtf(val);
}

__global__ void L2Backward(float *input, float *output,
                           float *grandOutput, float *grandInput) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  *(grandInput + i * blockDim.x + j)
    = (*(grandOutput + i)) * (*(input + i * blockDim.x + j)) / (*(output + i));
}


// Margin Ranking Loss function: max(0, Score)
__global__ void MarginRankingCriterionForward(
  float *inputPos, float *inputNeg, float *output) {
  int i = threadIdx.x;
  float src = 1.0 + inputPos[i] - inputNeg[i];
  output[i] = src <= 0 ? 0 : src;
}

__global__ void MarginRankingCriterionBackward(float *inputPos, float *inputNeg,
    float *gradInputPos, float *gradInputNeg) {
  int i = threadIdx.x;
  if (1.0 + inputPos[i] - inputNeg[i] > 0) {
    gradInputPos[i] = 1;
    gradInputNeg[i] = -1;
  } else {
    gradInputPos[i] = 0;
    gradInputNeg[i] = 0;
  }
}


// Debug version to get grad
__global__ void Debug(float *m_pos, float *m_neg, float *d_pos, float *d_neg,
  float *g_sub_pos, float *g_prd_pos, float *g_obj_pos, float *g_sub_neg, float *g_prd_neg, float *g_obj_neg) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  if (1.0 + *(d_pos + i) - *(d_neg + i) > 0) {
    *(g_sub_pos + i * blockDim.x + j) += *(m_pos + i * blockDim.x + j) / *(d_pos + i);
    *(g_prd_pos + i * blockDim.x + j) += *(m_pos + i * blockDim.x + j) / *(d_pos + i);
    *(g_obj_pos + i * blockDim.x + j) -= *(m_pos + i * blockDim.x + j) / *(d_pos + i);
  
    *(g_sub_neg + i * blockDim.x + j) -= *(m_neg + i * blockDim.x + j) / *(d_neg + i);
    *(g_prd_neg + i * blockDim.x + j) -= *(m_neg + i * blockDim.x + j) / *(d_neg + i);
    *(g_obj_neg + i * blockDim.x + j) += *(m_neg + i * blockDim.x + j) / *(d_neg + i);
  }
}


// Update Embedding
__global__ void UpdateEmbedding(float *des, int *idx, float *src, float lr) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  *(des + idx[i] * blockDim.x + j) -= *(src + i * blockDim.x + j) * lr;
}


__global__ void L2NormByIndex(float *m, int rows, int cols, int *idx, int sz, float *l2) {
  int threadId = threadIdx.x;
  if (threadId < sz) {
    int index = idx[threadId];
    if (index < rows) {
      float val = 0;
      for (int i = 0; i < cols; i++) {
        val += (*(m + index * cols + i)) * (*(m + index * cols + i));
      }
      *(l2 + threadId) = sqrtf(val);
    }
  }
}


__global__ void Normalize(float *m, int rows, int cols, int *idx, int sz, float *l2) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  if (i < sz) {
    int index = idx[i];
    if (index < rows && j < cols) {
      *(m + index * cols + j) /= *(l2 + i);
    }
  }
}


// Sum
__global__ void SumVector(float *v, int size, float *sum) {
  *sum = 0;
  for (int i = 0; i < size; i++) {
    *sum += v[i];
  }
}


// Test
__global__ void ResidualVectorT(int *sub, int *prd, int *obj, int size,
                                float *p_entities, float *p_relations,
                                float *residual_vector) {
  int i = blockIdx.y * gridDim.x + blockIdx.x;
  int j = threadIdx.x;
  if (i < size) {
    int sub_idx = *(sub + i);
    int prd_idx = *(prd + i);
    int obj_idx = *(obj + i);
  
    *(residual_vector + i * blockDim.x + j) = *(p_entities + sub_idx * blockDim.x + j)
      + *(p_relations + prd_idx * blockDim.x + j) - *(p_entities + obj_idx * blockDim.x + j);
  }
}

__global__ void L2NormT(float *input, int rows, int cols, float *output) {
  int i = blockIdx.y * gridDim.x + blockIdx.x;
  if (i < rows) {
    float val = 0;
    for (int j = 0; j < cols; j++) {
      val += (*(input + i * cols + j)) * (*(input + i * cols + j));
    }
    output[i] = sqrtf(val);
  }
}

__global__ void ResidualVectorFixedObjectT(int *sub, int *prd, int obj_idx,
      int size, float *p_entities, float *p_relations, float *residual_vector) {
  int i = blockIdx.y * gridDim.x + blockIdx.x;
  int j = threadIdx.x;
  if (i < size) {
    int sub_idx = *(sub + i);
    int prd_idx = *(prd + i); 
    *(residual_vector + i * blockDim.x + j) = *(p_entities + sub_idx * blockDim.x + j)
      + *(p_relations + prd_idx * blockDim.x + j) - *(p_entities + obj_idx * blockDim.x + j);
  }
}

__global__ void AccumulateT(float *count, float *benchmark, float *score, int size) {
  int i = blockIdx.y * gridDim.x + blockIdx.x;
  if (i < size && *(score + i) < *(benchmark + i)) {
    *(count + i) += 1.0;
  }
}

__global__ void FloatVectorSumT(float *input, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += *(input + i);
  }
  *input = sum;
}


class TransE {
private:
  int max_epochs;
  int print_epochs;
  int num_loops;
  int batch_size;
  float lr;

  DataLoader      *p_train_data_loader;
  ValidDataLoader *p_valid_data_loader;
  BatchDataLoader *p_batch_data_loader;
  
  LookupTable     *p_entities;
  LookupTable     *p_relations;

  string model_dir;

public:
  TransE(DataLoader*, ValidDataLoader*, BatchDataLoader*,
         LookupTable*, LookupTable*, int, int, float, string&);

  void train();
  void test();
  void save(int);
};

TransE::TransE(DataLoader *p_data, ValidDataLoader *p_valid, BatchDataLoader *p_batch,
               LookupTable *p_ent, LookupTable *p_rel, int e, int p, float l, string &dir) {
  p_train_data_loader = p_data;
  p_valid_data_loader = p_valid;
  p_batch_data_loader = p_batch;
  p_entities = p_ent;
  p_relations = p_rel;

  max_epochs = e;
  batch_size = p_train_data_loader->getBatchSize();
  num_loops = p_train_data_loader->getSize() / batch_size * max_epochs;
  print_epochs = p;

  lr = l;

  model_dir = dir;
  std::cout << "num_loops: " << num_loops << std::endl;
}

void TransE::save(int idx) {
  std::string entity_path = model_dir + "/entity_" + std::to_string(idx);
  p_entities->save(entity_path);
  std::string relation_path = model_dir + "/relation_" + std::to_string(idx);
  p_relations->save(relation_path);
}

void TransE::test() {
  int valid_num = p_valid_data_loader->size;
  int dimension = p_batch_data_loader->cols;
  dim3 blocksPerGrim(500, 100);
  float *residual_vector;
  cudaMalloc(&residual_vector, sizeof(float) * valid_num * dimension);
  float *residual_distance;
  cudaMalloc(&residual_distance, sizeof(float) * valid_num);

  ResidualVectorT<<<blocksPerGrim, dimension>>>(p_valid_data_loader->devSub,
    p_valid_data_loader->devPrd, p_valid_data_loader->devObj,
    valid_num, p_entities->d_table, p_relations->d_table, residual_vector);
  L2NormT<<<blocksPerGrim, 1>>>(residual_vector, valid_num, dimension, residual_distance);

  float *count;
  cudaMalloc(&count, sizeof(float) * valid_num);
  cudaMemset(count, 0, sizeof(float) * valid_num);
  float *rd_current;
  cudaMalloc(&rd_current, sizeof(float) * valid_num);
  
  for (int i = 1; i < p_entities->rows; i++) {
    ResidualVectorFixedObjectT<<<blocksPerGrim, dimension>>>(
      p_valid_data_loader->devSub, p_valid_data_loader->devPrd, i,
      valid_num, p_entities->d_table, p_relations->d_table, residual_vector);
    L2NormT<<<blocksPerGrim, 1>>>(residual_vector, valid_num, dimension, rd_current);
    AccumulateT<<<blocksPerGrim, 1>>>(count, residual_distance, rd_current, valid_num);
  }
  FloatVectorSumT<<<1, 1>>>(count, valid_num);
  float total_count;
  cudaMemcpy(&total_count, count, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "total count: " << total_count << std::endl;
  
  cudaFree(residual_vector);
  cudaFree(residual_distance);
  cudaFree(count);
  cudaFree(rd_current);
}

void TransE::train() {
  int batch_rows = p_batch_data_loader->rows;
  int batch_cols = p_batch_data_loader->cols;
  float print_epoch_error = 0.0, loop_error = 0.0;
  for (int i = 0; i < num_loops; i++) {
    p_train_data_loader->nextBatch(*p_batch_data_loader);
    p_batch_data_loader->lookuptable(p_entities, p_relations);

    // LinearForward starts
    float *m_pos, *m_neg_sub, *m_neg_prd, *m_neg_obj;
    cudaMalloc(&m_pos, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&m_neg_sub, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&m_neg_prd, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&m_neg_obj, sizeof(float) * batch_rows * batch_cols);
    LinearForward<<<batch_rows, batch_cols>>>(p_batch_data_loader->devSubEmbed,
      p_batch_data_loader->devPrdEmbed, p_batch_data_loader->devObjEmbed, m_pos);
    LinearForward<<<batch_rows, batch_cols>>>(p_batch_data_loader->devNSubEmbed,
      p_batch_data_loader->devPrdEmbed, p_batch_data_loader->devObjEmbed, m_neg_sub);
    LinearForward<<<batch_rows, batch_cols>>>(p_batch_data_loader->devSubEmbed,
      p_batch_data_loader->devNPrdEmbed, p_batch_data_loader->devObjEmbed, m_neg_prd);
    LinearForward<<<batch_rows, batch_cols>>>(p_batch_data_loader->devSubEmbed,
      p_batch_data_loader->devPrdEmbed, p_batch_data_loader->devNObjEmbed, m_neg_obj);

    // L2Forward starts
    float *distance_pos, *distance_neg_sub, *distance_neg_prd, *distance_neg_obj;
    cudaMalloc(&distance_pos, sizeof(float) * batch_rows);
    cudaMalloc(&distance_neg_sub, sizeof(float) * batch_rows);
    cudaMalloc(&distance_neg_prd, sizeof(float) * batch_rows);
    cudaMalloc(&distance_neg_obj, sizeof(float) * batch_rows);
    L2Forward<<<batch_rows, 1>>>(m_pos, distance_pos, batch_cols);
    L2Forward<<<batch_rows, 1>>>(m_neg_sub, distance_neg_sub, batch_cols);
    L2Forward<<<batch_rows, 1>>>(m_neg_prd, distance_neg_prd, batch_cols);
    L2Forward<<<batch_rows, 1>>>(m_neg_obj, distance_neg_obj, batch_cols);

    // MarginRankingCriterionForward starts
    float *score_nsub, *score_nprd, *score_nobj;
    cudaMalloc(&score_nsub, sizeof(float) * batch_rows);
    cudaMalloc(&score_nprd, sizeof(float) * batch_rows);
    cudaMalloc(&score_nobj, sizeof(float) * batch_rows);
    MarginRankingCriterionForward
      <<<1, batch_rows>>>(distance_pos, distance_neg_sub, score_nsub);
    MarginRankingCriterionForward
      <<<1, batch_rows>>>(distance_pos, distance_neg_prd, score_nprd);
    MarginRankingCriterionForward
      <<<1, batch_rows>>>(distance_pos, distance_neg_obj, score_nobj);

    {
    float *sum_nsub, *sum_nprd, *sum_nobj;
    cudaMalloc(&sum_nsub, sizeof(float));
    cudaMalloc(&sum_nprd, sizeof(float));
    cudaMalloc(&sum_nobj, sizeof(float));
    SumVector<<<1, 1>>>(score_nsub, batch_rows, sum_nsub);
    SumVector<<<1, 1>>>(score_nprd, batch_rows, sum_nprd);
    SumVector<<<1, 1>>>(score_nobj, batch_rows, sum_nobj);
    float h_sum_nsub, h_sum_nprd, h_sum_nobj;
    cudaMemcpy(&h_sum_nsub, sum_nsub, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_nprd, sum_nprd, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_nobj, sum_nobj, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sum_nsub);
    cudaFree(sum_nprd);
    cudaFree(sum_nobj);
    print_epoch_error += h_sum_nsub + h_sum_nprd + h_sum_nobj;
    loop_error += h_sum_nsub + h_sum_nprd + h_sum_nobj;
    
    // Epoch print
    if (i % print_epochs == 0) {
      printf("Epochs [%d] Loss [%f]\n", i, print_epoch_error / batch_size / print_epochs);
      print_epoch_error = 0;
    }

    // Loop print
    if (i != 0 && i % (p_train_data_loader->getSize() / batch_size) == 0) {
      int cur_loop_index = (int)((float)i / p_train_data_loader->getSize() * batch_size);
      float cur_loop_error = loop_error / p_train_data_loader->getSize();
      printf("Loops  [%d] Loss [%f]\n", cur_loop_index, cur_loop_error);
      loop_error = 0;
      // Evaluation
      // test();
    }

    // Write model
    if (i != 0 && i % (p_train_data_loader->getSize() / batch_size) == 0) {
      int cur_loop_index = (int)((float)i / p_train_data_loader->getSize() * batch_size);
      save(cur_loop_index);
    }
    }
    
    // MarginRankingCriterionBackward starts
    float *grad_distance_pos_nsub, *grad_distance_pos_nprd, *grad_distance_pos_nobj;
    float *grad_distance_neg_sub, *grad_distance_neg_prd, *grad_distance_neg_obj;
    cudaMalloc(&grad_distance_pos_nsub, sizeof(float) * batch_rows);
    cudaMalloc(&grad_distance_pos_nprd, sizeof(float) * batch_rows);
    cudaMalloc(&grad_distance_pos_nobj, sizeof(float) * batch_rows);
    cudaMalloc(&grad_distance_neg_sub, sizeof(float) * batch_rows);
    cudaMalloc(&grad_distance_neg_prd, sizeof(float) * batch_rows);
    cudaMalloc(&grad_distance_neg_obj, sizeof(float) * batch_rows);
    MarginRankingCriterionBackward<<<1, batch_rows>>>
      (distance_pos, distance_neg_sub, grad_distance_pos_nsub, grad_distance_neg_sub);
    MarginRankingCriterionBackward<<<1, batch_rows>>>
      (distance_pos, distance_neg_prd, grad_distance_pos_nprd, grad_distance_neg_prd);
    MarginRankingCriterionBackward<<<1, batch_rows>>>
      (distance_pos, distance_neg_obj, grad_distance_pos_nobj, grad_distance_neg_obj);


    // L2Backward starts
    float *grad_m_pos_nsub, *grad_m_pos_nprd, *grad_m_pos_nobj;
    float *grad_m_neg_sub, *grad_m_neg_prd, *grad_m_neg_obj;
    cudaMalloc(&grad_m_pos_nsub, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_m_pos_nprd, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_m_pos_nobj, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_m_neg_sub, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_m_neg_prd, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_m_neg_obj, sizeof(float) * batch_rows * batch_cols);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_pos, distance_pos, grad_distance_pos_nsub, grad_m_pos_nsub);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_pos, distance_pos, grad_distance_pos_nprd, grad_m_pos_nprd);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_pos, distance_pos, grad_distance_pos_nobj, grad_m_pos_nobj);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_neg_sub, distance_neg_sub, grad_distance_neg_sub, grad_m_neg_sub);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_neg_prd, distance_neg_prd, grad_distance_neg_prd, grad_m_neg_prd);
    L2Backward<<<batch_rows, batch_cols>>>
      (m_neg_obj, distance_neg_obj, grad_distance_neg_obj, grad_m_neg_obj);

    // LinearBackward starts
    float *grad_sub, *grad_prd, *grad_obj;
    float *grad_nsub, *grad_nprd, *grad_nobj;
    cudaMalloc(&grad_sub, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_prd, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_obj, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_nsub, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_nprd, sizeof(float) * batch_rows * batch_cols);
    cudaMalloc(&grad_nobj, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_sub, 0, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_prd, 0, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_obj, 0, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_nsub, 0, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_nprd, 0, sizeof(float) * batch_rows * batch_cols);
    cudaMemset((void*)grad_nobj, 0, sizeof(float) * batch_rows * batch_cols);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_pos_nsub, grad_sub, grad_prd, grad_obj);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_pos_nprd, grad_sub, grad_prd, grad_obj);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_pos_nobj, grad_sub, grad_prd, grad_obj);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_neg_sub, grad_nsub, grad_prd, grad_obj);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_neg_prd, grad_sub, grad_nprd, grad_obj);
    LinearBackward
      <<<batch_rows, batch_cols>>>(grad_m_neg_obj, grad_sub, grad_prd, grad_nobj);

    // Update
    float lr_cur = lr * (1 - (float)i / (num_loops + 1));
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_entities->d_table, p_batch_data_loader->devSub, grad_sub, lr_cur);
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_entities->d_table, p_batch_data_loader->devObj, grad_obj, lr_cur);
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_entities->d_table, p_batch_data_loader->devNSub, grad_nsub, lr_cur);
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_entities->d_table, p_batch_data_loader->devNObj, grad_nobj, lr_cur);
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_relations->d_table, p_batch_data_loader->devPrd, grad_prd, lr_cur);
    UpdateEmbedding<<<batch_rows, batch_cols>>>(
      p_relations->d_table, p_batch_data_loader->devNPrd, grad_nprd, lr_cur);


    std::set<int> ent_id_set;
    for (int j = 0; j < p_batch_data_loader->rows; j++) {
      ent_id_set.insert(p_batch_data_loader->sub[j]);
      ent_id_set.insert(p_batch_data_loader->obj[j]);
      ent_id_set.insert(p_batch_data_loader->nsub[j]);
      ent_id_set.insert(p_batch_data_loader->nobj[j]);
    }
    int h_entity_id_updated[ent_id_set.size()];
    int ptr = 0;
    for (int elem : ent_id_set) {
      h_entity_id_updated[ptr] = elem;
      ++ptr;
    }
    int *d_entity_id_updated;
    cudaMalloc(&d_entity_id_updated, sizeof(float) * ptr);
    cudaMemcpy(d_entity_id_updated, h_entity_id_updated, sizeof(float) * ptr, cudaMemcpyHostToDevice);
    float *l2_norm;
    cudaMalloc(&l2_norm, sizeof(float) * ptr);
    L2NormByIndex<<<1, ptr>>>(
      p_entities->d_table, p_entities->rows, p_entities->cols, d_entity_id_updated, ptr, l2_norm);
    Normalize<<<ptr, batch_cols>>>(
      p_entities->d_table, p_entities->rows, p_entities->cols, d_entity_id_updated, ptr, l2_norm);
    

    cudaFree(m_pos);
    cudaFree(m_neg_sub);
    cudaFree(m_neg_prd);
    cudaFree(m_neg_obj);
    cudaFree(distance_pos);
    cudaFree(distance_neg_sub);
    cudaFree(distance_neg_prd);
    cudaFree(distance_neg_obj);
    cudaFree(score_nsub);
    cudaFree(score_nprd);
    cudaFree(score_nobj);
    cudaFree(grad_distance_pos_nsub);
    cudaFree(grad_distance_pos_nprd);
    cudaFree(grad_distance_pos_nobj);
    cudaFree(grad_distance_neg_sub);
    cudaFree(grad_distance_neg_prd);
    cudaFree(grad_distance_neg_obj);
    cudaFree(grad_m_pos_nsub);
    cudaFree(grad_m_pos_nprd);
    cudaFree(grad_m_pos_nobj);
    cudaFree(grad_m_neg_sub);
    cudaFree(grad_m_neg_prd);
    cudaFree(grad_m_neg_obj);
    cudaFree(grad_sub);
    cudaFree(grad_prd);
    cudaFree(grad_obj);
    cudaFree(grad_nsub);
    cudaFree(grad_nprd);
    cudaFree(grad_nobj);
    cudaFree(d_entity_id_updated);
    cudaFree(l2_norm);
  }
}

#endif

