#pragma once

// Track A entry point: the explicit-workload gRPC/mTLS server
// (RemoteSumOfSquares, RemoteMatrixMultiply). Invoked from main.cpp for every
// command line that is not a Track B (`--foreground`) invocation.

void PrintWorkloadServerUsage(const char* executable);

// Parses the workload-server flags from argv, serves until shutdown, and
// returns the process exit code (0 ok, 1 startup failure, 2 usage error).
int RunWorkloadServer(int argc, char* argv[]);
