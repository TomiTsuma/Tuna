#pragma once

#include <string>

namespace tuna {

enum class TunaError {
    Ok,
    ConfigNotFound,
    ConfigParseError,
    LoggerInitFailed,
    DaemonAlreadyRunning,
    DaemonNotRunning,
};

std::string ToString(TunaError error);
bool IsError(TunaError error);

}  // namespace tuna
