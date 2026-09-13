#pragma once

namespace model_limits
{
    constexpr int MaxTransformerLayers = 75;
    constexpr int LayerNormCapacity = 2 * MaxTransformerLayers + 1;
    constexpr int FconnCapacity = 4 * MaxTransformerLayers + 1;
}
