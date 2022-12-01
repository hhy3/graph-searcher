#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace graph_searcher {

class DistanceTable {
public:
  DistanceTable(const float *q, int dim, int M, const float *pivots,
                const uint8_t *codes)
      : dim_(dim), M_(M), codes_(codes) {
    table_ = (float *)malloc(M * kK * sizeof(float));
    int ds = dim / M;
    std::memset(table_, 0, M * kK * sizeof(float));
    for (int chunk = 0; chunk < M; ++chunk) {
      float *cur_table = table_ + chunk * kK;
      for (int c = 0; c < kK; ++c) {
        const float *cur_pivot = pivots + c * dim;
        for (int i = chunk * ds; i < (chunk + 1) * ds; ++i) {
          float diff = q[i] - cur_pivot[i];
          cur_table[c] += diff * diff;
        }
      }
    }
  }

  float distance_to(int i) {
    float dist = 0.0;
    const uint8_t *code = codes_ + i * M_;
    for (int j = 0; j < M_; ++j) {
      float *cur_table = table_ + j * kK;
      dist += cur_table[code[j]];
    }
    return dist;
  }

private:
  constexpr static size_t kK = 256;
  int dim_, M_;
  float *table_;
  const uint8_t *codes_;
};

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

  DistanceTable get_dt(const float *q) const {
    return DistanceTable(q, dim_, M_, pivots_, codes_);
  }

  int size() const { return size_; }

  int M() const { return M_; }

  int dim() const { return dim_; }

private:
  constexpr static size_t kPageSize = 4096, kK = 256;
  int size_, M_, dim_;
  float *pivots_ = nullptr;
  uint8_t *codes_ = nullptr;
  float *dist_table_ = nullptr;
  float *centroid = nullptr;
};

} // namespace graph_searcher
