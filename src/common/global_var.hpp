

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <mcl/bn256.hpp>

#include "common/typedef.hpp"
using namespace mcl::bn;
#ifndef ZKCNN_GLOBAL_VAR_HPP
#define ZKCNN_GLOBAL_VAR_HPP

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::max;
using std::min;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::pair;
using std::make_pair;

extern bool g_debug;
typedef Fr F;
typedef G1 G;
#define F_ONE (Fr::one())
#define F_ZERO (Fr(0))

#define F_BYTE_SIZE 32

#endif
