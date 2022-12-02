#pragma once

#include <algorithm>

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
                             int l_search, int beamwidth = 1) {
  std::vector<bool> visited(graph.size());
  int ep = graph.ep();
  NeighborSet retset(l_search);
  std::vector<std::pair<int, float>> full_retset;
  const auto &pqtable = graph.pqtable();
  auto dt = pqtable.get_dt(q);
  retset.insert({ep, dt.distance_to(ep)});
  while (retset.has_next()) {
    std::vector<int> ids;
    while (ids.size() < beamwidth && retset.has_next()) {
      auto [u, dist] = retset.pop();
      ids.push_back(u);
    }
    std::vector<char *> bufs = graph.read(ids);
    for (int i = 0; i < ids.size(); ++i) {
      char *buf = bufs[i];
      int u = ids[i];
      int len = graph.degree(buf, u);
      int *edges = graph.edges(buf, u);
      full_retset.emplace_back(u, l2sqr(q, graph.data(buf, u), graph.dim()));
      for (int i = 0; i < len; ++i) {
        int v = edges[i];
        if (visited[v]) {
          continue;
        }
        visited[v] = true;
        float t_dist = dt.distance_to(v);
        retset.insert({v, t_dist});
      }
    }
  }
  std::sort(full_retset.begin(), full_retset.end(),
            [](auto &lhs, auto &rhs) { return lhs.second < rhs.second; });
  std::vector<int> ans(k);
  for (int i = 0; i < k; ++i) {
    ans[i] = full_retset[i].first;
  }
  return ans;
}

} // namespace graph_searcher
