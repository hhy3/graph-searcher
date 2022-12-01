#pragma once

#include <bits/stdc++.h>

#include "neighbor.hpp"
#include "utils.hpp"

namespace graph_searcher {

template <typename GraphType>
std::vector<int> greedy_search(GraphType &graph, const float *q, int k,
                               int l_search) {
  std::vector<bool> visited(graph.size());
  int ep = graph.ep();
  NeighborSet retset(l_search);
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

template <typename GraphType>
std::vector<int> beam_search(GraphType &graph, const float *q, int k,
                             int l_search) {
  std::vector<bool> visited(graph.size());
  int ep = graph.ep();
  NeighborSet retset(l_search);
  std::vector<std::pair<int, float>> full_retset;
  auto &pqtable = graph.pqtable();
  pqtable.set_query(q);
  retset.insert({ep, pqtable.distance_to(ep)});
  while (retset.has_next()) {
    auto [u, dist] = retset.pop();
    char *buf = graph.read({u})[0];
    int len = graph.degree(buf, u);
    int *edges = graph.edges(buf, u);
    std::vector<int> es;
    for (int i = 0; i < len; ++i) {
      es.push_back(edges[i]);
    }
    full_retset.emplace_back(u, l2sqr(q, graph.data(buf, u), graph.dim()));
    for (auto v : es) {
      // int v = edges[i];
      if (visited[v]) {
        continue;
      }
      visited[v] = true;
      char *buf = graph.read({v})[0];
      float cur_dist = l2sqr(q, graph.data(buf, v), graph.dim());
      float t_dist = pqtable.distance_to(v);
      std::cout << cur_dist << " " << t_dist << std::endl;
      retset.insert({v, t_dist});
    }
  }
  std::sort(full_retset.begin(), full_retset.end(),
            [](auto &lhs, auto &rhs) { return lhs.second < rhs.second; });
  std::vector<int> ans(k);
  for (int i = 0; i < k; ++i) {
    // std::cout << full_retset[i].second << ", ";
    ans[i] = full_retset[i].first;
  }
  // std::cout << "\n";
  return ans;
}

} // namespace graph_searcher