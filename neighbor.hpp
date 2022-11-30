#pragma once

#include <bits/stdc++.h>

namespace graph_searcher {

struct SimpleNeighbor {
  int id;
  float distance;

  SimpleNeighbor(int id = -1, float distance = float(1.0 / 0.0))
      : id(id), distance(distance) {}

  inline bool operator<(const SimpleNeighbor &other) const {
    return distance < other.distance;
  }
};

struct Neighbor {
  int id;
  float distance;
  bool checked;

  Neighbor(int id = -1, float distance = float(1.0 / 0.0), bool checked = false)
      : id(id), distance(distance), checked(checked) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

class HeapNeighborSet {
public:
  HeapNeighborSet(size_t open_capacity, size_t closed_capacity)
      : open_capacity_(open_capacity), closed_capacity_(closed_capacity) {}

  void insert(const SimpleNeighbor &nbr) {
    if (open_.size() < open_capacity_) {
      open_.emplace(nbr);
    } else if (nbr < *open_.rbegin()) {
      open_.erase(std::prev(open_.end()));
      open_.insert(nbr);
    }
  }

  SimpleNeighbor pop() {
    auto ret = *open_.begin();
    if (closed_.size() < closed_capacity_) {
      closed_.insert(ret);
    } else if (ret < *closed_.rbegin()) {
      closed_.erase(std::prev(closed_.end()));
      closed_.insert(ret);
    }
    open_.erase(ret);
    return ret;
  }

  bool has_next() { return !open_.empty(); }

  std::vector<int> get_topk(int k) {
    std::vector<int> ans(k);
    int i = 0;
    for (auto it = closed_.begin(); i < k; it++, i++) {
      ans[i] = it->id;
    }
    return ans;
  }

private:
  size_t open_capacity_, closed_capacity_;
  std::set<SimpleNeighbor> open_, closed_;
};

class NeighborSet {
public:
  explicit NeighborSet(size_t capacity = 0)
      : size_(0), capacity_(capacity), data_(capacity_ + 1) {}

  void insert(SimpleNeighbor nbr) {
    if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
      return;
    }
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (data_[mid].distance > nbr.distance) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
    data_[lo] = {nbr.id, nbr.distance, false};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
  }

  SimpleNeighbor pop() {
    data_[cur_].checked = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].checked) {
      cur_++;
    }
    return {data_[pre].id, data_[pre].distance};
  }

  bool has_next() const { return cur_ < size_; }

  std::vector<int> get_topk(int k) {
    std::vector<int> ans(k);
    for (int i = 0; i < k; ++i) {
      ans[i] = data_[i].id;
    }
    return ans;
  }

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

} // namespace graph_searcher