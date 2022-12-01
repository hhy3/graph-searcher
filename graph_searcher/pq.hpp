#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace graph_searcher {

class PQTable {
public:
  PQTable() = default;

  explicit PQTable(int dim) : dim_(dim) {}

  void load_compressed(const std::string &filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.read((char *)&size_, 4);
    reader.read((char *)&M_, 4);
    codes_ = (uint8_t *)malloc(size_ * M_);
    reader.read((char *)codes_, size_ * M_);
  }

  void load_pivots(const std::string &filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.seekg(4096 + 8, reader.cur);
    pivots_ = (float *)aligned_alloc(kPageSize, kK * dim_ * sizeof(float));
    reader.read((char *)pivots_, kK * dim_ * sizeof(float));
    reader.seekg(8, reader.cur);
    centroid = (float *)malloc(dim_ * sizeof(float));
    reader.read((char *)centroid, dim_ * sizeof(float));
  }

  void set_query(const float *q) {
    if (!dist_table_) {
      dist_table_ = (float *)malloc(M_ * kK * sizeof(float));
    }
    int ds = dim_ / M_;
    std::memset(dist_table_, 0, M_ * kK * sizeof(float));
    for (int chunk = 0; chunk < M_; ++chunk) {
      float *cur_table = dist_table_ + chunk * kK;
      for (int c = 0; c < kK; ++c) {
        float *cur_pivot = pivots_ + c * dim_;
        for (int i = chunk * ds; i < chunk * (ds + 1); ++i) {
          float diff = q[i] - cur_pivot[i];
          cur_table[c] += diff * diff;
        }
      }
    }
  }

  float distance_to(int i) {
    float dist = 0.0;
    uint8_t *code = codes_ + i * M_;
    for (int j = 0; j < M_; ++j) {
      float *cur_table = dist_table_ + j * kK;
      dist += cur_table[code[j]];
    }
    return dist;
  }

  int size() { return size_; }

  int M() { return M_; }

  int dim() { return dim_; }

private:
  constexpr static size_t kPageSize = 4096, kK = 256;
  int size_, M_, dim_;
  float *pivots_ = nullptr;
  uint8_t *codes_ = nullptr;
  float *dist_table_ = nullptr;
  float *centroid = nullptr;
};

} // namespace graph_searcher
