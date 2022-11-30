#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

namespace graph_searcher {

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

} // namespace graph_searcher