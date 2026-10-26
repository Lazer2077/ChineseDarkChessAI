// ChineseDarkChessAI: probe-dispersion allocation for hidden-identity chance nodes.
// GPL-3.0. The native backbone is separately attributed MIT-licensed jqe.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cdca {
struct Allocation {
    std::array<int, 6> order{};
    double mean = 0, deviation = 0;
    bool active = false;
};
// Rank identities by their contribution p_i (v_i - E[v])^2 to probe variance.
// This is a search-allocation heuristic, not a confidence bound or equilibrium result.
inline Allocation allocate(const int* count, const double* value, int n, double threshold) {
    if (n < 1 || n > 6 || threshold < 0) throw std::invalid_argument("invalid chance allocation");
    Allocation result;
    double total = 0;
    for (int i = 0; i < n; ++i) {
        if (count[i] <= 0 || !std::isfinite(value[i])) throw std::invalid_argument("invalid outcome");
        total += count[i];result.order[i] = i;
    }
    for (int i = 0; i < n; ++i) result.mean += count[i] / total * value[i];
    std::array<double, 6> influence{};
    double variance = 0;
    for (int i = 0; i < n; ++i) {
        influence[i] = count[i] / total * std::pow(value[i] - result.mean, 2);
        variance += influence[i];
    }
    result.deviation = std::sqrt(variance);
    result.active = n > 1 && result.deviation > threshold;
    if (result.active) std::stable_sort(result.order.begin(), result.order.begin() + n,
        [&](int a, int b) { return influence[a] > influence[b]; });
    return result;
}
}
