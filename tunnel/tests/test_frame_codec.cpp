#include <catch2/catch_test_macros.hpp>

#include "tuna/tunnel/frame_codec.h"

using namespace tuna;
using namespace tuna::tunnel;

namespace {

Frame MakeFrame(uint64_t id, const std::string& payload) {
    Frame frame;
    frame.mutable_header()->set_type(FRAME_TYPE_GPU);
    frame.mutable_header()->set_frame_id(id);
    frame.mutable_header()->set_channel(1);
    frame.set_payload(payload);
    return frame;
}

}  // namespace

TEST_CASE("Frame codec round-trips one frame", "[tunnel][codec]") {
    std::string wire;
    REQUIRE(EncodeFrame(MakeFrame(42, "hello"), &wire));

    FrameDecoder decoder;
    REQUIRE(decoder.Feed(wire));
    Frame out;
    REQUIRE(decoder.Next(&out));
    CHECK(out.header().frame_id() == 42);
    CHECK(out.payload() == "hello");
    CHECK_FALSE(decoder.Next(&out));
}

TEST_CASE("Frame codec reassembles a frame fed one byte at a time", "[tunnel][codec]") {
    std::string wire;
    REQUIRE(EncodeFrame(MakeFrame(7, std::string(1000, 'x')), &wire));

    FrameDecoder decoder;
    Frame out;
    for (size_t i = 0; i + 1 < wire.size(); ++i) {
        REQUIRE(decoder.Feed(std::string_view(&wire[i], 1)));
        REQUIRE_FALSE(decoder.Next(&out));
    }
    REQUIRE(decoder.Feed(std::string_view(&wire.back(), 1)));
    REQUIRE(decoder.Next(&out));
    CHECK(out.payload().size() == 1000);
}

TEST_CASE("Frame codec splits coalesced frames in order", "[tunnel][codec]") {
    std::string wire;
    for (uint64_t id = 1; id <= 3; ++id) {
        REQUIRE(EncodeFrame(MakeFrame(id, "p" + std::to_string(id)), &wire));
    }
    FrameDecoder decoder;
    REQUIRE(decoder.Feed(wire));
    for (uint64_t id = 1; id <= 3; ++id) {
        Frame out;
        REQUIRE(decoder.Next(&out));
        CHECK(out.header().frame_id() == id);
    }
}

TEST_CASE("Frame codec rejects a length prefix over the limit", "[tunnel][codec]") {
    // 0xFFFFFFFF little-endian: a hostile peer asking us to buffer 4 GiB.
    const std::string wire("\xFF\xFF\xFF\xFF", 4);
    FrameDecoder decoder;
    CHECK_FALSE(decoder.Feed(wire));
    CHECK_FALSE(decoder.Feed("more"));  // stays failed
}

TEST_CASE("Frame codec rejects an unparseable frame body", "[tunnel][codec]") {
    std::string wire("\x03\x00\x00\x00", 4);
    wire += "\xFF\xFF\xFF";  // not a valid protobuf message
    FrameDecoder decoder;
    CHECK_FALSE(decoder.Feed(wire));
}

TEST_CASE("Frame codec refuses to encode an oversized frame", "[tunnel][codec]") {
    std::string wire = "keep";
    CHECK_FALSE(EncodeFrame(MakeFrame(1, std::string(kMaxFrameBytes, 'z')), &wire));
    CHECK(wire == "keep");
}
