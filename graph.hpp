#pragma once

#include <bits/stdc++.h>

namespace graph_searcher {

struct NSGGraph {
  char *buf;

  explicit NSGGraph(const std::string &graph_path,
                    const std::string &data_path) {
    std::ifstream graph_reader(graph_path, std::ios::binary);
    std::ifstream data_reader(data_path, std::ios::binary);
    data_reader.read((char *)&dim_, 4);
    vec_nbytes = dim_ * sizeof(float);
    data_reader.seekg(0, data_reader.end);
    size_t data_bytes = data_reader.tellg();
    data_reader.seekg(0, data_reader.beg);
    size_ = data_bytes / (vec_nbytes + 4);
    graph_reader.seekg(0, graph_reader.end);
    size_t nbytes = graph_reader.tellg();
    graph_reader.seekg(0, graph_reader.beg);
    int width;
    graph_reader.read((char *)&width, 4);
    graph_reader.read((char *)&ep_, 4);
    node_nbytes = vec_nbytes + (width + 1) * sizeof(int);
    buf = (char *)aligned_alloc(4096, node_nbytes * size_);
    size_t bytes_read = 8;
    for (int i = 0; i < size_; ++i) {
      data_reader.seekg(4, data_reader.cur);
      data_reader.read((char *)data(i), vec_nbytes);
      int k;
      graph_reader.read((char *)&k, 4);
      *(int *)(get_bytes(i) + vec_nbytes) = k;
      graph_reader.read((char *)edges(i), k * 4);
    }
  }

  const float *data(int i) const { return (const float *)get_bytes(i); }

  const int degree(int i) const { return *(int *)(get_bytes(i) + vec_nbytes); }

  int *edges(int i) const {
    return (int *)(get_bytes(i) + vec_nbytes + sizeof(int));
  }

  const int size() const { return size_; }

  const int ep() const { return ep_; }

  const int dim() const { return dim_; }

private:
  int size_;
  int dim_;
  int ep_;

  int node_nbytes;
  int vec_nbytes;

  char *get_bytes(int i) const { return buf + i * node_nbytes; }
};

struct DiskannGraph {
  char *buf;
  explicit DiskannGraph(const std::string &path) {
    std::ifstream reader(path.c_str(), std::ios::binary);
    reader.seekg(0, reader.end);
    size_t nbytes = reader.tellg();
    reader.seekg(0, reader.beg);
    buf = (char *)aligned_alloc(kSectorLen, nbytes);
    reader.read(buf, nbytes);
    size_ = *(int64_t *)(buf + 8);
    dim_ = *(int64_t *)(buf + 16);
    ep_ = *(int64_t *)(buf + 24);
    node_nbytes = *(int64_t *)(buf + 32);
    nnodes_per_sector = *(int64_t *)(buf + 40);
    vec_nbytes = dim_ * sizeof(float);
    std::cout << "done reading diskann, nbytes: " << nbytes
              << ", size: " << size_ << ", dim: " << dim_ << ", ep: " << ep_
              << std::endl;
  }

  const float *data(int i) const { return (const float *)get_bytes(i); }

  const int degree(int i) const { return *(int *)(get_bytes(i) + vec_nbytes); }

  int *edges(int i) const {
    return (int *)(get_bytes(i) + vec_nbytes + sizeof(int));
  }

  const int size() const { return size_; }

  const int ep() const { return ep_; }

  const int dim() const { return dim_; }

private:
  constexpr static size_t kSectorLen = 4096;
  int size_;
  int dim_;
  int ep_;
  int nnodes_per_sector;
  int vec_nbytes;
  int node_nbytes;
  char *get_bytes(int i) const {
    return buf + (i / nnodes_per_sector + 1) * kSectorLen +
           (i % nnodes_per_sector) * node_nbytes;
  }
};

} // namespace graph_searcher