#include <catch2/catch_test_macros.hpp>
#include "tuna_frames.pb.h"

// Wire protocol v1 — see docs/plans/2026-09-23-tuna-execution-plan.md §4.

TEST_CASE("Frame round-trips through serialize/parse", "[proto][tunnel]") {
    tuna::Frame original;
    original.mutable_header()->set_type(tuna::FRAME_TYPE_HEARTBEAT);
    original.mutable_header()->set_frame_id(42);
    original.mutable_header()->set_channel(0);
    original.mutable_header()->set_timestamp_ns(1234567890123456789ULL);
    original.set_payload("test-payload-bytes");

    std::string wire;
    REQUIRE(original.SerializeToString(&wire));

    tuna::Frame parsed;
    REQUIRE(parsed.ParseFromString(wire));

    REQUIRE(parsed.header().type() == tuna::FRAME_TYPE_HEARTBEAT);
    REQUIRE(parsed.header().frame_id() == 42);
    REQUIRE(parsed.header().channel() == 0);
    REQUIRE(parsed.header().timestamp_ns() == 1234567890123456789ULL);
    REQUIRE(parsed.payload() == "test-payload-bytes");
}

TEST_CASE("frame_id survives values above the 32-bit range", "[proto][tunnel]") {
    tuna::Frame f;
    const uint64_t big = 0x1'0000'0001ULL;  // would wrap in a uint32 field
    f.mutable_header()->set_frame_id(big);

    std::string wire;
    REQUIRE(f.SerializeToString(&wire));
    tuna::Frame parsed;
    REQUIRE(parsed.ParseFromString(wire));
    REQUIRE(parsed.header().frame_id() == big);
}

TEST_CASE("RpcRequest round-trips", "[proto][rpc]") {
    tuna::RpcRequest req;
    req.set_call_id(7);
    req.set_api_id(101);
    req.set_flags(0x3);  // ASYNC | HAS_BULK
    req.set_thread_id(0xABCDEF01);
    req.set_args(std::string("\x01\x02\x03\x04", 4));
    req.set_bulk_ref(555);

    std::string wire;
    REQUIRE(req.SerializeToString(&wire));
    tuna::RpcRequest parsed;
    REQUIRE(parsed.ParseFromString(wire));

    REQUIRE(parsed.call_id() == 7);
    REQUIRE(parsed.api_id() == 101);
    REQUIRE(parsed.flags() == 0x3);
    REQUIRE(parsed.thread_id() == 0xABCDEF01);
    REQUIRE(parsed.args() == std::string("\x01\x02\x03\x04", 4));
    REQUIRE(parsed.bulk_ref() == 555);
}

TEST_CASE("Hello carries device capabilities", "[proto][control]") {
    tuna::Hello hello;
    hello.set_protocol_version(1);
    hello.set_client_os("windows");
    hello.set_cuda_driver_version(12040);
    auto* dev = hello.add_devices();
    dev->set_name("NVIDIA RTX 4090");
    dev->set_total_mem(25769803776ULL);  // 24 GiB
    dev->set_cc_major(8);
    dev->set_cc_minor(9);

    std::string wire;
    REQUIRE(hello.SerializeToString(&wire));
    tuna::Hello parsed;
    REQUIRE(parsed.ParseFromString(wire));

    REQUIRE(parsed.protocol_version() == 1);
    REQUIRE(parsed.client_os() == "windows");
    REQUIRE(parsed.cuda_driver_version() == 12040);
    REQUIRE(parsed.devices_size() == 1);
    REQUIRE(parsed.devices(0).name() == "NVIDIA RTX 4090");
    REQUIRE(parsed.devices(0).total_mem() == 25769803776ULL);
    REQUIRE(parsed.devices(0).cc_major() == 8);
    REQUIRE(parsed.devices(0).cc_minor() == 9);
}
