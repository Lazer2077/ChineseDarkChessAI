// Read-only C ABI around the unchanged MIT jqe static NNUE evaluator.
#include "nnue.h"
#include "position.h"
#include <exception>
#include <string>

namespace { thread_local std::string error; }

extern "C" const char* cdca_load_nnue(const char* path) {
    try {
        error.clear();
        if (!jq::nnue::load(path, &error)) return error.c_str();
        jq::nnue::set_quantized(true);
        return nullptr;
    } catch (const std::exception& ex) { error = ex.what(); return error.c_str(); }
}

extern "C" const char* cdca_eval_observer_fen(const char* fen, int* value) {
    try {
        error.clear();
        if (!jq::nnue::loaded()) { error = "NNUE not loaded"; return error.c_str(); }
        jq::Position position;
        if (!position.set_fen(fen, &error)) return error.c_str();
        *value = jq::nnue::evaluate(position);
        return nullptr;
    } catch (const std::exception& ex) { error = ex.what(); return error.c_str(); }
}
