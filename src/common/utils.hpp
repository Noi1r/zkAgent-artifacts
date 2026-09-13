#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include <gkr/circuit.h>

char ceilPow2BitLength(u32 n);

void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
              const F &alpha, const F &beta,int thd=32);

void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init,int thread=32);

void initLayer(layer &circuit, long size, layerType ty);

#endif
