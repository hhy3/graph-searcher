#pragma once

#include <bits/stdc++.h>

#include "neighbor.hpp"
#include "utils.hpp"

namespace graph_searcher {

template <typename GraphType>
std::vector<int> greedy_search(const GraphType &graph, const float *q, int k,
                        int l_search) {
  std::vector<bool> visited(graph.size());
  int ep = graph.ep();
  // NeighborSet retset(l_search);
  HeapNeighborSet retset(l_search, k);
  retset.insert({ep, l2sqr(q, graph.data(ep), graph.dim())});
  while (retset.has_next()) {
    auto [u, dist] = retset.pop();
    int len = graph.degree(u);
    int *edges = graph.edges(u);
    for (int i = 0; i < len; ++i) {
      int v = edges[i];
      if (visited[v]) {
        continue;
      }
      visited[v] = true;
      float cur_dist = l2sqr(q, graph.data(v), graph.dim());
      retset.insert({v, cur_dist});
    }
  }
  return retset.get_topk(k);
}

} // namespace graph_searcher