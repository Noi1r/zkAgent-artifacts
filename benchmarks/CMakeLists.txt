function(zkagent_benchmark target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} gpt_lib mcl mclbn256 OpenMP::OpenMP_CXX)
endfunction()

zkagent_benchmark(benchmark_zkembed_preprocess benchmark_zkembed_preprocess.cpp)
zkagent_benchmark(benchmark_zkembed benchmark_zkembed.cpp)
zkagent_benchmark(benchmark_zkposenc main_benchmark_zkposenc.cpp)

zkagent_benchmark(benchmark_transformer main_benchmark_transformer.cpp)
zkagent_benchmark(benchmark_inference main_benchmark_transformer.cpp)
target_compile_definitions(benchmark_inference PRIVATE ZKAGENT_TABLE6_DEFAULT=1)

add_executable(demo_zkdecode_run main_benchmark_zkdecode.cpp)
target_link_libraries(demo_zkdecode_run gpt_lib mcl OpenMP::OpenMP_CXX)
