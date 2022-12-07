#pragma once

#include <fstream>
#include <immintrin.h>
#include <iostream>

namespace graph_searcher {

using dist_func = float(const void *, const void *, const size_t);

float l2sqrbf16(const void *X, const void *Y, const size_t d) {
  const float *x = (const float *)X;
  const uint8_t *y = (const uint8_t *)Y;
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      y += 16;
      auto t = _mm256_sub_ps(xx, (__m256)yy);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
    }
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      y += 16;
      auto t = _mm256_sub_ps(xx, (__m256)yy);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

float l2sqrfp16(const void *X, const void *Y, const size_t d) {
  const float *x = (const float *)X;
  const uint8_t *y = (const uint8_t *)Y;
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      auto yy = _mm256_cvtph_ps(zz);
      y += 16;
      auto t = _mm256_sub_ps(xx, yy);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
    }
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      auto yy = _mm256_cvtph_ps(zz);
      y += 16;
      auto t = _mm256_sub_ps(xx, yy);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

float l2sqr(const void *X, const void *Y, const int d) {
  const float *x = (const float *)X;
  const float *y = (const float *)Y;
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto yy = _mm256_loadu_ps(y);
      y += 8;
      auto t = _mm256_sub_ps(xx, yy);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
    }
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto yy = _mm256_loadu_ps(y);
      y += 8;
      auto t = _mm256_sub_ps(xx, yy);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
  __m256 sum = _mm256_setzero_ps();
}

struct FP16Computer {
  float operator()(const float *q, const uint8_t *code, const int d) {
    return l2sqrfp16(q, code, d);
  }
};

struct FlatComputer {
  float operator()(const float *q, const float *code, const int d) {
    return l2sqr(q, code, d);
  }
};

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