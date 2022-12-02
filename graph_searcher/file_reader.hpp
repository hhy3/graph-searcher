#pragma once

#include <fcntl.h>
#include <iostream>
#include <liburing.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace graph_searcher {

class IOContext {
public:
  IOContext() { io_uring_queue_init(kMaxnr, &ring_, 0); }

  ~IOContext() { io_uring_queue_exit(&ring_); }

  io_uring &ring() { return ring_; }

private:
  io_uring ring_;
  constexpr static size_t kMaxnr = 32;
};

struct ReadRequest {
  size_t offset;
  size_t len;
  char *buf;
  ReadRequest() = default;
  ReadRequest(size_t offset, size_t len, char *buf)
      : offset(offset), len(len), buf(buf) {}
};

class FileReader {
public:
  FileReader(const std::string &filename) {
    fd_ = open(filename.c_str(), O_DIRECT | O_RDONLY);
  }

  void read(const std::vector<ReadRequest> &reqs) {
    std::unique_lock<std::mutex> lk(mtx_);
    auto &ctx = ctx_map_[std::this_thread::get_id()];
    lk.unlock();
    read_impl(reqs, ctx);
  }

private:
  int fd_;
  std::unordered_map<std::thread::id, IOContext> ctx_map_;
  IOContext ctx_;
  std::mutex mtx_;

  void read_impl(const std::vector<ReadRequest> &reqs, IOContext &ctx) {
    for (const auto &req : reqs) {
      auto sqe = io_uring_get_sqe(&ctx_.ring());
      io_uring_prep_read(sqe, fd_, req.buf, req.len, req.offset);
    }
    io_uring_submit(&ctx_.ring());
    io_uring_cqe *cqe;
    for (int i = 0; i < reqs.size(); ++i) {
      io_uring_wait_cqe(&ctx_.ring(), &cqe);
      io_uring_cqe_seen(&ctx_.ring(), cqe);
    }
  }
};

} // namespace graph_searcher