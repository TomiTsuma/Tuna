# Shared Track B transport + RPC runtime (tunnel/), used by both the guest
# agent and tuna-server. Include after cmake/TunaProtobuf.cmake.
#
# Provides:
#   tuna_tunnel           static library
#   tuna_rpc_bench        Phase 2 latency/throughput benchmark
#   tuna_add_tunnel_tests()  call inside a BUILD_TESTING block once Catch2 is
#                            available; a second call (superbuild) is a no-op.
include_guard(GLOBAL)

find_package(Threads REQUIRED)

set(TUNA_TUNNEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../tunnel")
cmake_path(NORMAL_PATH TUNA_TUNNEL_DIR)

add_library(tuna_tunnel STATIC
    ${TUNA_TUNNEL_DIR}/src/frame_codec.cpp
    ${TUNA_TUNNEL_DIR}/src/tcp_socket.cpp
    ${TUNA_TUNNEL_DIR}/src/transport.cpp
    ${TUNA_TUNNEL_DIR}/src/session.cpp
    ${TUNA_TUNNEL_DIR}/src/rpc.cpp
)
target_include_directories(tuna_tunnel PUBLIC ${TUNA_TUNNEL_DIR}/include)
target_link_libraries(tuna_tunnel PUBLIC tuna_proto Threads::Threads)
if(WIN32)
    target_link_libraries(tuna_tunnel PUBLIC ws2_32)
    # inet_pton and friends need Vista+ headers under MinGW.
    target_compile_definitions(tuna_tunnel PRIVATE _WIN32_WINNT=0x0601)
endif()
target_compile_options(tuna_tunnel PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)

add_executable(tuna_rpc_bench ${TUNA_TUNNEL_DIR}/bench/rpc_echo_bench.cpp)
target_link_libraries(tuna_rpc_bench PRIVATE tuna_tunnel)

add_executable(tuna_ping ${TUNA_TUNNEL_DIR}/tools/tuna_ping.cpp)
target_link_libraries(tuna_ping PRIVATE tuna_tunnel)

function(tuna_add_tunnel_tests)
    if(TARGET tuna_tunnel_tests)
        return()
    endif()
    add_executable(tuna_tunnel_tests
        ${TUNA_TUNNEL_DIR}/tests/test_frame_codec.cpp
        ${TUNA_TUNNEL_DIR}/tests/test_rpc_loopback.cpp
    )
    target_link_libraries(tuna_tunnel_tests PRIVATE Catch2::Catch2WithMain tuna_tunnel)
    catch_discover_tests(tuna_tunnel_tests TEST_PREFIX "tunnel: ")
    # Keeps the benchmark binary working in CI without asserting timings.
    add_test(NAME tuna_rpc_bench_smoke COMMAND tuna_rpc_bench --calls 500 --posts 5000)
endfunction()
