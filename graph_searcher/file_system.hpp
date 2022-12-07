#pragma once

#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <libaio.h>
#include <liburing.h>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace graph_searcher {

struct ReadRequest {
  size_t offset;
  size_t len;
  char *buf;
  ReadRequest() = default;
  ReadRequest(size_t offset, size_t len, char *buf)
      : offset(offset), len(len), buf(buf) {}
};

class RandomAccessFileReader {
public:
  virtual void read(const std::vector<ReadRequest> &reqs, int fd) {
    for (const auto &req : reqs) {
      ::pread(fd, req.buf, req.len, req.offset);
    }
  }

  virtual ~RandomAccessFileReader() = default;
};

class AIOReader : public RandomAccessFileReader {
public:
  AIOReader() {
    if (io_setup(kMaxnr, &ctx_)) {
      std::cerr << "io_setup failed\n";
      exit(1);
    }
  }

  ~AIOReader() { io_destroy(ctx_); }

  void read(const std::vector<ReadRequest> &reqs, int fd) override {
    int n = reqs.size();
    std::vector<iocb *> cbs(n);
    std::vector<io_event> events(n);
    std::vector<iocb> cb(n);
    for (int i = 0; i < n; ++i) {
      io_prep_pread(&cb[i], fd, reqs[i].buf, reqs[i].len, reqs[i].offset);
    }
    for (int i = 0; i < n; ++i) {
      cbs[i] = &cb[i];
    }
    if (io_submit(ctx_, n, cbs.data()) != n) {
      std::cerr << "io_submit failed\n";
      exit(1);
    }
    if (io_getevents(ctx_, n, n, events.data(), nullptr) != n) {
      std::cerr << "io_getevents failed\n";
      exit(1);
    }
  }

private:
  io_context_t ctx_ = 0;
  constexpr static size_t kMaxnr = 32;
};

class IOUringReader : public RandomAccessFileReader {
public:
  IOUringReader() { io_uring_queue_init(kMaxnr, &ring_, 0); }

  ~IOUringReader() { io_uring_queue_exit(&ring_); }

  void read(const std::vector<ReadRequest> &reqs, int fd) override {
    for (const auto &req : reqs) {
      auto sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, fd, req.buf, req.len, req.offset);
    }
    io_uring_submit(&ring_);
    io_uring_cqe *cqe;
    for (int i = 0; i < reqs.size(); ++i) {
      io_uring_wait_cqe(&ring_, &cqe);
      io_uring_cqe_seen(&ring_, cqe);
    }
  }

private:
  io_uring ring_;
  constexpr static size_t kMaxnr = 32;
};

class FileSystem {
public:
  FileSystem(const std::string &filename) {
    fd_ = open(filename.c_str(), O_DIRECT | O_RDONLY);
    start_ = std::chrono::high_resolution_clock::now();
  }

  void read(const std::vector<ReadRequest> &reqs) {
    std::unique_lock<std::mutex> lk(mtx_);
    io_cnt_ += reqs.size();
    auto &reader = reader_map_[std::this_thread::get_id()];
    if (reader == nullptr) {
      reader = std::make_unique<AIOReader>();
    }
    lk.unlock();
    reader->read(reqs, fd_);
  }

  ~FileSystem() {
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start_).count();
    std::cout << "IOPS: " << io_cnt_ / elapsed << std::endl;
  }

private:
  int fd_;
  std::unordered_map<std::thread::id, std::unique_ptr<RandomAccessFileReader>>
      reader_map_;
  std::mutex mtx_;
  int64_t io_cnt_ = 0;
  decltype(std::chrono::high_resolution_clock::now()) start_;
};

} // namespace graph_searcher