#pragma once

#include <fcntl.h>
#include <liburing.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <iostream>

namespace graph_searcher {

namespace {
constexpr size_t kMaxnr = 32;
}

class IOContext {
public:
  IOContext(size_t maxnr) { io_uring_queue_init(maxnr, &ring_, 0); }

  ~IOContext() { io_uring_queue_exit(&ring_); }

  io_uring &ring() { return ring_; }

private:
  io_uring ring_;
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
  FileReader(const std::string &filename) : ctx_(kMaxnr) {
    fd_ = open(filename.c_str(), O_DIRECT | O_RDONLY);
  }

  void read(const std::vector<ReadRequest> &reqs) {
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

private:
  int fd_;
  IOContext ctx_;
};

} // namespace graph_searcher