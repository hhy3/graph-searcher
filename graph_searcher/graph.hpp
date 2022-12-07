#pragma once

#include <cstdlib>
#include <mutex>
#include <thread>

#include "file_system.hpp"
#include "pq.hpp"
#include "utils.hpp"
#include <cassert>
#include <immintrin.h>

namespace graph_searcher {

template <typename T, typename Cal> struct Vec2d {
  using value_type = T;
  int n, d, codesize;
  uint8_t *data = nullptr;
  Cal computer;
  Vec2d() = default;
  Vec2d(int n, int d, int codesize)
      : n(n), d(d), codesize(codesize),
        data((uint8_t *)aligned_alloc(64, n * d * sizeof(value_type))) {}
  Vec2d(const Vec2d &) = delete;
  Vec2d &operator=(const Vec2d &) = delete;
  Vec2d(Vec2d &&rhs)
      : n(rhs.n), d(rhs.d), codesize(rhs.codesize), data(rhs.data) {
    rhs.data = nullptr;
  }
  Vec2d &operator=(Vec2d &&rhs) {
    n = rhs.n;
    d = rhs.d;
    data = rhs.data;
    codesize = rhs.codesize;
    rhs.data = nullptr;
    return *this;
  }

  T *operator[](int i) { return (T *)(data + i * codesize); }

  float dist_to(const float *q, int i) { return computer(q, operator[](i), d); }

  ~Vec2d() { free(data); }
};

template <typename Cal>
Vec2d<uint8_t, FP16Computer> SQFP16_transform(Vec2d<float, Cal> &vec) {
  int n = vec.n, d = vec.d;
  Vec2d<uint8_t, FP16Computer> ret(n, d, 2 * d);
  for (int i = 0; i < n; ++i) {
    float *vec_from = vec[i];
    uint8_t *code_to = ret[i];
    for (int j = 0; j < d; j += 8) {
      auto x = _mm256_loadu_ps(vec_from + j);
      auto y = _mm256_cvtps_ph(x, 0);
      _mm_storeu_si128((__m128i *)(code_to + j * 2), y);
    }
  }
  return ret;
}

struct Graph {
  int n, m;
  std::vector<int> edges;

  Graph() = default;

  Graph(int n, int m) : n(n), m(m), edges(n * (1 + m)) {}

  void add_edge(int u, int v) {
    int offset = u * (1 + m);
    int &deg = edges[offset];
    edges[offset + 1 + deg] = v;
    deg++;
  }

  int *edge_list(int i) { return &edges[i * (1 + m) + 1]; }

  int &degree(int i) { return edges[i * (1 + m)]; }
};

struct NSG {
  int dim_, size_, ep_;
  Graph graph;
  Vec2d<float, FlatComputer> dataset;
  Vec2d<uint8_t, FP16Computer> codes;
  NSG(const std::string &graph_path, const std::string &data_path) {
    std::ifstream graph_reader(graph_path, std::ios::binary);
    std::ifstream data_reader(data_path, std::ios::binary);
    data_reader.read((char *)&dim_, 4);

    int vec_nbytes = dim_ * sizeof(float);
    data_reader.seekg(0, data_reader.end);
    size_t data_bytes = data_reader.tellg();
    data_reader.seekg(0, data_reader.beg);
    size_ = data_bytes / (vec_nbytes + 4);
    dataset = decltype(dataset)(size_, dim_, dim_ * sizeof(float));
    graph_reader.seekg(0, graph_reader.end);
    size_t nbytes = graph_reader.tellg();
    graph_reader.seekg(0, graph_reader.beg);
    int width;
    graph_reader.read((char *)&width, 4);
    graph_reader.read((char *)&ep_, 4);
    graph = Graph(size_, width);
    for (int i = 0; i < size_; ++i) {
      data_reader.seekg(4, data_reader.cur);
      data_reader.read((char *)data(i), dim_ * sizeof(float));
      int k;
      graph_reader.read((char *)&k, 4);
      graph.degree(i) = k;
      graph_reader.read((char *)edges(i), k * 4);
    }
    codes = SQFP16_transform(dataset);
  }

  float dist_to(const float *q, int i) { return dataset.dist_to(q, i); }

  float *data(int i) { return dataset[i]; }

  int *edges(int i) { return graph.edge_list(i); }

  int degree(int i) { return graph.degree(i); }

  int size() const { return size_; }

  int ep() const { return ep_; }

  int dim() const { return dim_; }
};

struct NSGGraph {
  char *buf;

  dist_func dist;

  NSGGraph(const std::string &graph_path, const std::string &data_path) {
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
    if (buf == nullptr) {
      std::cerr << "allocate memory failed\n";
      exit(1);
    }
    for (int i = 0; i < size_; ++i) {
      data_reader.seekg(4, data_reader.cur);
      data_reader.read((char *)data(i), vec_nbytes_);
      int k;
      graph_reader.read((char *)&k, 4);
      *(int *)(get_bytes(i) + vec_nbytes_) = k;
      graph_reader.read((char *)edges(i), k * 4);
    }
    std::cout << "before" << std::endl;
    transform_bf16();
    std::cout << "after" << std::endl;
  }

  void FloatToBFloat16(const float *src, void *dst, int64_t size) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
    uint16_t *q = reinterpret_cast<uint16_t *>(dst);
    for (; size != 0; p += 2, q++, size--) {
      *q = p[1];
    }
  }

  void BFloat16ToFloat(const void *src, float *dst, int64_t size) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
    uint16_t *q = reinterpret_cast<uint16_t *>(dst);
    for (; size != 0; p++, q += 2, size--) {
      q[0] = 0;
      q[1] = *p;
    }
  }

  void transform_bf16() {
    int64_t new_vec_nbytes_ = vec_nbytes_ / 2;
    int64_t new_node_nbytes_ = node_nbytes_ + (new_vec_nbytes_ - vec_nbytes_);
    char *new_buf = (char *)aligned_alloc(4096, new_node_nbytes_ * size_);

#pragma omp parallel for schedule(static) num_threads(8)
    for (int64_t i = 0; i < size_; ++i) {
      char *from = get_bytes(i);
      char *to = new_buf + i * new_node_nbytes_;
      float *vec_from = (float *)from;
      uint8_t *code_to = (uint8_t *)to;
      FloatToBFloat16(vec_from, code_to, dim_);

      std::memcpy(to + new_vec_nbytes_, from + vec_nbytes_,
                  (node_nbytes_ - vec_nbytes_));
    }
    vec_nbytes_ = new_vec_nbytes_;
    node_nbytes_ = new_node_nbytes_;
    free(buf);
    buf = new_buf;
  }

  void transform_fp16() {
    int64_t new_vec_nbytes_ = vec_nbytes_ / 2;
    int64_t new_node_nbytes_ = node_nbytes_ + (new_vec_nbytes_ - vec_nbytes_);
    char *new_buf = (char *)aligned_alloc(4096, new_node_nbytes_ * size_);

#pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < size_; ++i) {
      char *from = get_bytes(i);
      char *to = new_buf + i * new_node_nbytes_;
      float *vec_from = (float *)from;
      uint8_t *code_to = (uint8_t *)to;
      for (int j = 0; j < dim_; j += 8) {
        auto x = _mm256_loadu_ps(vec_from + j);
        auto y = _mm256_cvtps_ph(x, 0);
        _mm_storeu_si128((__m128i *)(code_to + j * 2), y);
      }
      std::memcpy(to + new_vec_nbytes_, from + vec_nbytes_,
                  (node_nbytes_ - vec_nbytes_));
    }
    vec_nbytes_ = new_vec_nbytes_;
    node_nbytes_ = new_node_nbytes_;
    free(buf);
    buf = new_buf;
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
  int size_, dim_, ep_;
  int64_t node_nbytes_, vec_nbytes_;

  char *get_bytes(int64_t i) const { return buf + i * node_nbytes_; }
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
      : index_prefix_(index_prefix), file_(disk_index_path(index_prefix_)) {
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
    file_.read(reqs);
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

  PQTable &pqtable() { return pqtable_; }

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
  FileSystem file_;
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