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

# TLS 1.3 + mTLS (Phase 2b) uses OpenSSL >= 3.0 on every platform.
# Linux: libssl-dev. Windows/MSVC: vcpkg or the Shining Light installer.
# Windows/MinGW: FindOpenSSL looks for MinGW-named libraries, but the
# Shining Light "OpenSSL-Win64" install only ships MSVC import libraries. Those
# link fine for OpenSSL's C API, so point FindOpenSSL at them.
if(MINGW AND NOT LIB_EAY)
    set(_tuna_openssl_root "$ENV{ProgramFiles}/OpenSSL-Win64")
    if(OPENSSL_ROOT_DIR)
        set(_tuna_openssl_root "${OPENSSL_ROOT_DIR}")
    endif()
    if(EXISTS "${_tuna_openssl_root}/lib/VC/x64/MD/libcrypto.lib")
        set(OPENSSL_ROOT_DIR "${_tuna_openssl_root}")
        set(LIB_EAY "${_tuna_openssl_root}/lib/VC/x64/MD/libcrypto.lib" CACHE FILEPATH "")
        set(SSL_EAY "${_tuna_openssl_root}/lib/VC/x64/MD/libssl.lib" CACHE FILEPATH "")
    endif()
endif()
find_package(OpenSSL 3.0 REQUIRED)

set(TUNA_TUNNEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../tunnel")
cmake_path(NORMAL_PATH TUNA_TUNNEL_DIR)

add_library(tuna_tunnel STATIC
    ${TUNA_TUNNEL_DIR}/src/frame_codec.cpp
    ${TUNA_TUNNEL_DIR}/src/tcp_socket.cpp
    ${TUNA_TUNNEL_DIR}/src/transport.cpp
    ${TUNA_TUNNEL_DIR}/src/session.cpp
    ${TUNA_TUNNEL_DIR}/src/rpc.cpp
    ${TUNA_TUNNEL_DIR}/src/tls.cpp
    ${TUNA_TUNNEL_DIR}/src/pki.cpp
    ${TUNA_TUNNEL_DIR}/src/client.cpp
)
target_include_directories(tuna_tunnel PUBLIC ${TUNA_TUNNEL_DIR}/include)
target_link_libraries(tuna_tunnel PUBLIC tuna_proto Threads::Threads OpenSSL::SSL OpenSSL::Crypto)
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
        ${TUNA_TUNNEL_DIR}/tests/test_bulk.cpp
        ${TUNA_TUNNEL_DIR}/tests/test_tls.cpp
    )
    target_link_libraries(tuna_tunnel_tests PRIVATE Catch2::Catch2WithMain tuna_tunnel)
    catch_discover_tests(tuna_tunnel_tests TEST_PREFIX "tunnel: ")
    # Keeps the benchmark binary working in CI without asserting timings.
    add_test(NAME tuna_rpc_bench_smoke COMMAND tuna_rpc_bench --calls 500 --posts 5000 --bulk-mb 4)
    add_test(NAME tuna_rpc_bench_tls_smoke COMMAND tuna_rpc_bench --tls --calls 500 --posts 5000 --bulk-mb 4)
endfunction()
