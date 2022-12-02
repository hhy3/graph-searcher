#include <omp.h>
#include <unordered_set>

#include "graph_searcher/graph.hpp"
#include "graph_searcher/search.hpp"
#include "graph_searcher/utils.hpp"

using namespace graph_searcher;

int main(int argc, char **argv) {
  const char *graph_type = argv[1];
  const char *graph_path = argv[2];
  const char *data_path = argv[3];
  const char *query_path = argv[4];
  const char *gt_path = argv[5];
  int k = std::stoi(argv[6]);
  int ef = std::stoi(argv[7]);
  int num_threads = std::stoi(argv[8]);
  float *query;
  int *gt;
  int nq, dim, gt_k;
  load_fvecs(query_path, query, nq, dim);
  load_fvecs(gt_path, gt, nq, gt_k);

  auto do_iter = [&](int iters, auto &&f, auto &&...args) {
    for (int iter = 0; iter < iters; ++iter) {
      std::cout << "iter: " << iter << std::endl;
      f(std::forward<decltype(args)>(args)...);
    }
  };

  auto run = [&](auto &&graph, auto &&f, auto &&...args) {
    int64_t cnt = 0;
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < nq; ++i) {
      auto ret = f(std::forward<decltype(graph)>(graph), query + i * dim, k,
                   std::forward<decltype(args)>(args)...);
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

  if (std::string(graph_type) == "vamana") {
    VamanaGraph graph(graph_path);
    do_iter(10, run, graph, greedy_search<decltype(graph)>, ef);
  } else if (std::string(graph_type) == "nsg") {
    NSGGraph graph(graph_path, data_path);
    do_iter(10, run, graph, greedy_search<decltype(graph)>, ef);
  } else if (std::string(graph_type) == "diskvamana") {
    DiskVamanaGraph graph(graph_path);
    do_iter(10, run, graph, beam_search<decltype(graph)>, ef, 4);
  }
}