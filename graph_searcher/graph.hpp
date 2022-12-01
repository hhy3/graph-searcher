#pragma once

#include <bits/stdc++.h>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "file_reader.hpp"
#include "pq.hpp"

namespace graph_searcher {

struct NSGGraph {
  char *buf;

  explicit NSGGraph(const std::string &graph_path,
                    const std::string &data_path) {
    std::ifstream graph_reader(graph_path, std::ios::binary);
    std::ifstream data_reader(data_path, std::ios::binary);
    data_reader.read((char *)&dim_, 4);
    vec_nbytes_ = dim_ * sizeof(float);
    data_reader.seekg(0, data_reader.end);
    size_t data_bytes = data_reader.tellg();
    data_reader.seekg(0, data_reader.beg);
    size_ = data_bytes / (vec_nbytes_ + 4);
    graph_reader.seekg(0, graph_reader.end);
    size_t nbytes = graph_reader.tellg();
    graph_reader.seekg(0, graph_reader.beg);
    int width;
    graph_reader.read((char *)&width, 4);
    graph_reader.read((char *)&ep_, 4);
    node_nbytes_ = vec_nbytes_ + (width + 1) * sizeof(int);
    buf = (char *)aligned_alloc(4096, node_nbytes_ * size_);
    size_t bytes_read = 8;
    for (int i = 0; i < size_; ++i) {
      data_reader.seekg(4, data_reader.cur);
      data_reader.read((char *)data(i), vec_nbytes_);
      int k;
      graph_reader.read((char *)&k, 4);
      *(int *)(get_bytes(i) + vec_nbytes_) = k;
      graph_reader.read((char *)edges(i), k * 4);
    }
  }

  float *data(int i) { return (float *)get_bytes(i); }

  int degree(int i) const { return *(int *)(get_bytes(i) + vec_nbytes_); }

  int *edges(int i) {
    return (int *)(get_bytes(i) + vec_nbytes_ + sizeof(int));
  }

  int size() const { return size_; }

  int ep() const { return ep_; }

  int dim() const { return dim_; }

private:
  int size_, dim_, ep_, node_nbytes_, vec_nbytes_;

  char *get_bytes(int i) const { return buf + i * node_nbytes_; }
};

struct VamanaGraph {
  char *buf;
  explicit VamanaGraph(const std::string &path) {
    std::ifstream reader(path.c_str(), std::ios::binary);
    reader.seekg(0, reader.end);
    size_t nbytes = reader.tellg();
    reader.seekg(0, reader.beg);
    buf = (char *)aligned_alloc(kSectorLen, nbytes);
    reader.read(buf, nbytes);
    size_ = *(int64_t *)(buf + 8);
    dim_ = *(int64_t *)(buf + 16);
    ep_ = *(int64_t *)(buf + 24);
    node_nbytes_ = *(int64_t *)(buf + 32);
    nnodes_per_sector_ = *(int64_t *)(buf + 40);
    vec_nbytes_ = dim_ * sizeof(float);
    std::cout << "done reading diskann, nbytes: " << nbytes
              << ", size: " << size_ << ", dim: " << dim_ << ", ep: " << ep_
              << std::endl;
  }

  float *data(int i) { return (float *)get_bytes(i); }

  int degree(int i) const { return *(int *)(get_bytes(i) + vec_nbytes_); }

  int *edges(int i) {
    return (int *)(get_bytes(i) + vec_nbytes_ + sizeof(int));
  }

  int size() const { return size_; }

  int ep() const { return ep_; }

  int dim() const { return dim_; }

private:
  constexpr static size_t kSectorLen = 4096;
  int size_, dim_, ep_, nnodes_per_sector_, vec_nbytes_, node_nbytes_;
  char *get_bytes(int i) const {
    return buf + (i / nnodes_per_sector_ + 1) * kSectorLen +
           (i % nnodes_per_sector_) * node_nbytes_;
  }
};

struct DiskVamanaGraph {
  DiskVamanaGraph(const std::string &index_prefix)
      : index_prefix_(index_prefix), reader_(disk_index_path(index_prefix_)) {
    std::ifstream meta_reader(disk_index_path(index_prefix));
    constexpr size_t kMetaLen = 48;
    char buf[kMetaLen];
    meta_reader.read(buf, kMetaLen);
    size_ = *(int64_t *)(buf + 8);
    dim_ = *(int64_t *)(buf + 16);
    vec_nbytes_ = dim_ * sizeof(float);
    ep_ = *(int64_t *)(buf + 24);
    node_nbytes_ = *(int64_t *)(buf + 32);
    nnodes_per_sector_ = *(int64_t *)(buf + 40);
    pqtable_ = PQTable(dim_);
    pqtable_.load_compressed(pq_compressed_path(index_prefix));
    pqtable_.load_pivots(pq_pivots_path(index_prefix));
    std::cout << "DiskVamanaGraph init done" << std::endl;
  }

  std::vector<char *> read(const std::vector<int> &ids) {
    char *buf = get_buffer();
    int n = ids.size();
    std::vector<char *> bufs(n);
    std::vector<ReadRequest> reqs(n);
    for (int i = 0; i < n; ++i) {
      reqs[i] = ReadRequest(sector_no(ids[i]) * kPageSize, kPageSize,
                            buf + i * kPageSize);
      bufs[i] = reqs[i].buf;
    }
    reader_.read(reqs);
    return bufs;
  }

  char *get_buffer() {
    std::unique_lock<std::mutex> lk(mtx_);
    auto id = std::this_thread::get_id();
    char *&p = page_buffer_[id];
    lk.unlock();
    if (!p) {
      p = (char *)aligned_alloc(kPageSize, kPageSize * kPagesPerThread);
    }
    return p;
  }

  PQTable &pqtable() {
    return pqtable_;
  }

  float *data(char *buf, int i) { return (float *)get_node(buf, i); }

  int degree(char *buf, int i) const {
    return *(int *)(get_node(buf, i) + vec_nbytes_);
  }

  int *edges(char *buf, int i) {
    return (int *)(get_node(buf, i) + vec_nbytes_ + sizeof(int));
  }

  int size() const { return size_; }

  int ep() const { return ep_; }

  int dim() const { return dim_; }

private:
  constexpr static size_t kPageSize = 4096;
  constexpr static size_t kPagesPerThread = 128;
  std::string index_prefix_;
  std::unordered_map<std::thread::id, char *> page_buffer_;
  FileReader reader_;
  std::mutex mtx_;

  PQTable pqtable_;

  int size_, dim_, M_, ep_, vec_nbytes_, node_nbytes_, nnodes_per_sector_;

  std::string disk_index_path(const std::string &prefix) const {
    return prefix + "_disk.index";
  }

  std::string pq_pivots_path(const std::string &prefix) const {
    return prefix + "_pq_pivots.bin";
  }

  std::string pq_compressed_path(const std::string &prefix) const {
    return prefix + "_pq_compressed.bin";
  }

  size_t sector_no(int i) const { return i / nnodes_per_sector_ + 1; }

  char *get_node(char *buf, int i) const {
    return buf + (i % nnodes_per_sector_) * node_nbytes_;
  }
};

} // namespace graph_searcher