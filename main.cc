#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

float l2sqr(const float *x, const float *y, const int d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx1 = _mm256_loadu_ps(x);
    x += 8;
    auto yy1 = _mm256_loadu_ps(y);
    y += 8;
    auto t1 = _mm256_sub_ps(xx1, yy1);
    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t1, t1));
    auto xx2 = _mm256_loadu_ps(x);
    x += 8;
    auto yy2 = _mm256_loadu_ps(y);
    y += 8;
    auto t2 = _mm256_sub_ps(xx2, yy2);
    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t2, t2));
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
  __m256 sum = _mm256_setzero_ps();
}

template <typename T>
void load_fvecs(const char *filename, T *&p, int &n, int &dim) {
  std::ifstream fs(filename, std::ios::binary);
  fs.read((char *)&dim, 4);
  fs.seekg(0, std::ios::end);
  n = fs.tellg() / (4 + dim * sizeof(T));
  fs.seekg(0, std::ios::beg);
  std::cout << "Read path: " << filename << ", nx: " << n << ", dim: " << dim
            << std::endl;
  p = reinterpret_cast<T *>(aligned_alloc(64, n * dim * sizeof(T)));
  for (int i = 0; i < n; ++i) {
    fs.seekg(4, std::ios::cur);
    fs.read((char *)&p[i * dim], dim * sizeof(T));
  }
}

struct Neighbor {
  unsigned id;
  float distance;
  bool checked;

  Neighbor(unsigned id = -1, float distance = float(1.0 / 0.0),
           bool checked = false)
      : id(id), distance(distance), checked(checked) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

class NeighborSet {
public:
  explicit NeighborSet(size_t capacity = 0)
      : size_(0), capacity_(capacity), data_(capacity_ + 1) {}

  void insert(const Neighbor &nbr) {
    if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
      return;
    }
    size_t p = std::lower_bound(data_.begin(), data_.begin() + size_, nbr) -
               data_.begin();
    std::memmove(&data_[p + 1], &data_[p], (size_ - p) * sizeof(Neighbor));
    data_[p] = nbr;
    if (size_ < capacity_) {
      size_++;
    }
    if (p < cur_) {
      cur_ = p;
    }
  }

  Neighbor pop() {
    data_[cur_].checked = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].checked) {
      cur_++;
    }
    return data_[pre];
  }

  bool has_next() const { return cur_ < size_; }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  Neighbor &operator[](size_t i) { return data_[i]; }

  const Neighbor &operator[](size_t i) const { return data_[i]; }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

private:
  size_t size_, capacity_, cur_;
  std::vector<Neighbor> data_;
};

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
    std::cout << "size: " << size_ << std::endl;
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

template <typename GraphType>
std::vector<int> search(const GraphType &graph, const float *q, int k,
                        int l_search) {
  std::vector<bool> visited(graph.size());
  int ep = graph.ep();
  NeighborSet retset(l_search);
  retset.insert(Neighbor(ep, l2sqr(q, graph.data(ep), graph.dim())));
  while (retset.has_next()) {
    auto [u, dist, _] = retset.pop();
    int len = graph.degree(u);
    int *edges = graph.edges(u);
    for (int i = 0; i < len; ++i) {
      int v = edges[i];
      if (visited[v]) {
        continue;
      }
      visited[v] = true;
      float cur_dist = l2sqr(q, graph.data(v), graph.dim());
      retset.insert(Neighbor(v, cur_dist));
    }
  }
  std::vector<int> ans(k);
  for (int i = 0; i < k; ++i) {
    ans[i] = retset[i].id;
  }
  return ans;
}

int main(int argc, char **argv) {
  const char *graph_type = argv[1];
  const char *graph_path = argv[2];
  const char *data_path = argv[3];
  const char *query_path = argv[4];
  const char *gt_path = argv[5];
  int k = std::stoi(argv[6]);
  int ef = std::stoi(argv[7]);
  float *query;
  int *gt;
  int nq, dim, gt_k;
  load_fvecs(query_path, query, nq, dim);
  load_fvecs(gt_path, gt, nq, gt_k);

  auto run = [&](const auto &graph) {
    int64_t cnt = 0;
    auto st = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for num_threads(8)
    for (int i = 0; i < nq; ++i) {
      auto ret = search(graph, query + i * dim, k, ef);
      std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + k);
      for (auto x : ret) {
        if (st.count(x)) {
          cnt++;
        }
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(ed - st).count();
    std::cout << "Recall: " << double(cnt) / (nq * k) << std::endl;
    std::cout << "QPS: " << double(nq) / elapsed << std::endl;
  };

  if (std::string(graph_type) == "diskann") {
    DiskannGraph graph(graph_path);
    run(graph);
  } else if (std::string(graph_type) == "nsg") {
    NSGGraph graph(graph_path, data_path);
    run(graph);
  }
}