set(gpt_src
    model/attention_softmax.cpp
    gkr/circuit.cpp
    common/compiled_mle.cpp
    common/global_var.cpp
    common/mle.cpp
    model/gpt2.cpp
    model/linear_witness.cpp
    gkr/gkr_proof.cpp
    gkr/gkr_verifier.cpp
    pcs/hyrax.cpp
    gkr/input_commitment.cpp
    gkr/input_reduction_link.cpp
    gkr/input_routing_compiled.cpp
    gkr/input_tensor_view.cpp
    lookup/lookup_bundle.cpp
    lookup/lookup_input_link.cpp
    lookup/lookup_input_routing_compiled.cpp
    lookup/lookup_witness.cpp
    pcs/mvkzg.cpp
    model/nonlinear_quant.cpp
    common/polynomial.cpp
    gkr/prover.cpp
    model/quant_plan.cpp
    lookup/logup_gkr.cpp
    lookup/shout.cpp
    pcs/srs.cpp
    common/timer.cpp
    common/utils.cpp
    gkr/verifier.cpp
    gkr/wiring_compiled.cpp
    zkdecode/zkdecode_proof.cpp
    zkdecode/zkdecode_prover.cpp
    zkdecode/zkdecode_output_prover.cpp
    zkdecode/zkdecode_output_verifier.cpp
    zkdecode/zkdecode_exp_digits.cpp
    zkdecode/zkdecode_arithmetic_prover.cpp
    zkdecode/zkdecode_arithmetic_verifier.cpp
    zkdecode/zkdecode_input_link.cpp
    zkdecode/zkdecode_lookup.cpp
    zkdecode/zkdecode_verifier.cpp
    zkembed/zkembed.cpp
    zkposenc/zkposenc.cpp
)
add_library(gpt_lib ${gpt_src})
target_link_libraries(gpt_lib OpenMP::OpenMP_CXX)
target_compile_definitions(gpt_lib PUBLIC MCL_USE_OMP)

target_include_directories(gpt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(generate_tokens apps/generate_tokens.cpp)
target_link_libraries(generate_tokens gpt_lib mcl mclbn256 OpenMP::OpenMP_CXX)

add_executable(warmup_srs apps/warmup_srs.cpp)
target_link_libraries(warmup_srs gpt_lib mcl OpenMP::OpenMP_CXX)
