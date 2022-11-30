#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <queue>

#include "graph.hpp"
#include "search.hpp"
#include "utils.hpp"

using namespace graph_searcher;

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

  auto do_iters = [&](int iters, auto &f, auto &&...args) {
    for (int iter = 0; iter < iters; ++iter) {
      std::cout << "iter: " << iter << std::endl;
      f(args...);
    }
  };

  auto run = [&](const auto &graph) {
    int64_t cnt = 0;
    auto st = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for num_threads(8)
    for (int i = 0; i < nq; ++i) {
      auto ret = greedy_search(graph, query + i * dim, k, ef);
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
    do_iters(10, run, graph);
  } else if (std::string(graph_type) == "nsg") {
    NSGGraph graph(graph_path, data_path);
    do_iters(10, run, graph);
  }
}