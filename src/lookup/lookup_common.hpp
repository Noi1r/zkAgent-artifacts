#ifndef ZKAGENT_LOOKUP_COMMON_HPP
#define ZKAGENT_LOOKUP_COMMON_HPP

#include "common/global_var.hpp"

#include <vector>

struct LookupReadClaim
{
    std::vector<G1> commitments;
    std::vector<Fr> point;
    Fr evaluation = Fr(0);
};

#endif
