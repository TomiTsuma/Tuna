#include "core/error_handler.h"

namespace tuna {

std::string ToString(TunaError error) {
    switch (error) {
        case TunaError::Ok: return "Ok";
        case TunaError::ConfigNotFound: return "ConfigNotFound";
        case TunaError::ConfigParseError: return "ConfigParseError";
        case TunaError::LoggerInitFailed: return "LoggerInitFailed";
        case TunaError::DaemonAlreadyRunning: return "DaemonAlreadyRunning";
        case TunaError::DaemonNotRunning: return "DaemonNotRunning";
    }
    return "Unknown";
}

bool IsError(TunaError error) {
    return error != TunaError::Ok;
}

}  // namespace tuna
