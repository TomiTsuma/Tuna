#include "remoting/worker_process.h"

#include <cerrno>
#include <cstring>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace tuna {

namespace {

constexpr size_t kHeaderBytes = 4 + sizeof(tunnel::SessionToken);

void SetError(std::string* error, const std::string& message) {
    if (error) *error = message;
}

std::string EncodeHeader(const HandoffHeader& header) {
    std::string bytes(kHeaderBytes, '\0');
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>((header.kind >> (8 * i)) & 0xff);
    std::memcpy(bytes.data() + 4, header.token.data(), header.token.size());
    return bytes;
}

HandoffHeader DecodeHeader(const char* bytes) {
    HandoffHeader header;
    for (int i = 0; i < 4; ++i) header.kind |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[i])) << (8 * i);
    std::memcpy(header.token.data(), bytes + 4, header.token.size());
    return header;
}

#ifdef _WIN32
std::wstring Widen(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
    return wide;
}

// Quotes one argument the way CommandLineToArgvW / the CRT parse it.
std::wstring QuoteArg(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos) return arg;
    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
        } else if (c == L'"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted += c;
            backslashes = 0;
        } else {
            quoted.append(backslashes, L'\\');
            quoted += c;
            backslashes = 0;
        }
    }
    quoted.append(backslashes * 2, L'\\');
    quoted += L'"';
    return quoted;
}

// One Job object for the listener's lifetime: when the listener exits, for any
// reason, Windows kills every worker in it.
HANDLE WorkerJob() {
    static HANDLE job = [] {
        HANDLE h = CreateJobObjectW(nullptr, nullptr);
        if (h) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(h, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
        }
        return h;
    }();
    return job;
}

bool ReadFull(HANDLE handle, char* data, size_t size) {
    while (size > 0) {
        DWORD got = 0;
        if (!ReadFile(handle, data, static_cast<DWORD>(size), &got, nullptr) || got == 0) return false;
        data += got;
        size -= got;
    }
    return true;
}

bool WriteFull(HANDLE handle, const char* data, size_t size) {
    while (size > 0) {
        DWORD put = 0;
        if (!WriteFile(handle, data, static_cast<DWORD>(size), &put, nullptr)) return false;
        data += put;
        size -= put;
    }
    return true;
}
#endif

}  // namespace

// ------------------------------------------------------------ WorkerProcess

#ifdef _WIN32

std::unique_ptr<WorkerProcess> WorkerProcess::Spawn(const std::string& executable,
                                                    const std::vector<std::string>& args, std::string* error) {
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE stdin_read = nullptr, stdin_write = nullptr, ack_read = nullptr, ack_write = nullptr;
    if (!CreatePipe(&stdin_read, &stdin_write, &inheritable, 0) ||
        !CreatePipe(&ack_read, &ack_write, &inheritable, 0)) {
        SetError(error, "CreatePipe failed: " + std::to_string(GetLastError()));
        return nullptr;
    }
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(ack_read, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             &inheritable, OPEN_EXISTING, 0, nullptr);

    // Inherit exactly these handles. Winsock sockets are inheritable by
    // default, and the listening socket must never reach a worker.
    HANDLE inherit[] = {stdin_read, ack_write, nul};
    SIZE_T list_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &list_size);
    std::vector<char> list_storage(list_size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_storage.data());
    InitializeProcThreadAttributeList(list, 1, 0, &list_size);
    UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof(inherit), nullptr,
                              nullptr);

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdin_read;
    startup.StartupInfo.hStdOutput = nul;  // workers log to their own file
    startup.StartupInfo.hStdError = nul;
    startup.lpAttributeList = list;

    std::wstring command = QuoteArg(Widen(executable));
    for (const std::string& arg : args) command += L" " + QuoteArg(Widen(arg));
    command += L" --ack-handle " + std::to_wstring(reinterpret_cast<uintptr_t>(ack_write));

    PROCESS_INFORMATION info{};
    // Suspended until it is in the Job, so it can never run outside it.
    const BOOL created = CreateProcessW(Widen(executable).c_str(), command.data(), nullptr, nullptr, TRUE,
                                        CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
                                        nullptr, nullptr, &startup.StartupInfo, &info);
    const DWORD create_error = GetLastError();
    DeleteProcThreadAttributeList(list);
    CloseHandle(stdin_read);
    CloseHandle(ack_write);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!created) {
        CloseHandle(stdin_write);
        CloseHandle(ack_read);
        SetError(error, "CreateProcess failed: " + std::to_string(create_error));
        return nullptr;
    }
    if (HANDLE job = WorkerJob()) AssignProcessToJobObject(job, info.hProcess);
    ResumeThread(info.hThread);
    CloseHandle(info.hThread);

    std::unique_ptr<WorkerProcess> worker(new WorkerProcess());
    worker->pid_ = info.dwProcessId;
    worker->process_ = info.hProcess;
    worker->to_worker_ = stdin_write;
    worker->from_worker_ = ack_read;
    return worker;
}

WorkerProcess::~WorkerProcess() {
    if (process_ && !Exited()) StopAndWait(std::chrono::seconds(1));
    CloseChannel();
    if (from_worker_) CloseHandle(from_worker_);
    if (process_) CloseHandle(process_);
}

bool WorkerProcess::HandOff(const HandoffHeader& header, tunnel::TcpStream stream,
                            std::chrono::milliseconds timeout) {
    if (!to_worker_) return false;
    WSAPROTOCOL_INFOW info{};
    if (WSADuplicateSocketW(static_cast<SOCKET>(stream.native()), pid_, &info) != 0) return false;
    std::string message = EncodeHeader(header);
    message.append(reinterpret_cast<const char*>(&info), sizeof(info));
    if (!WriteFull(to_worker_, message.data(), message.size())) return false;
    // Keep our copy of the socket open until the worker has created its own
    // descriptor from the protocol info; `stream` closes it on return.
    return ReadAck(timeout);
}

bool WorkerProcess::ReadAck(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        DWORD available = 0;
        if (!PeekNamedPipe(from_worker_, nullptr, 0, nullptr, &available, nullptr)) return false;  // worker gone
        if (available > 0) {
            char ack = 0;
            return ReadFull(from_worker_, &ack, 1) && ack == 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool WorkerProcess::Exited(int* code) {
    if (!exit_code_ && WaitForSingleObject(process_, 0) == WAIT_OBJECT_0) {
        DWORD value = 0;
        GetExitCodeProcess(process_, &value);
        exit_code_ = static_cast<int>(value);
    }
    if (exit_code_ && code) *code = *exit_code_;
    return exit_code_.has_value();
}

void WorkerProcess::CloseChannel() {
    if (to_worker_) {
        CloseHandle(to_worker_);
        to_worker_ = nullptr;
    }
}

int WorkerProcess::StopAndWait(std::chrono::milliseconds timeout) {
    CloseChannel();
    if (WaitForSingleObject(process_, static_cast<DWORD>(timeout.count())) != WAIT_OBJECT_0) {
        TerminateProcess(process_, 1);
        WaitForSingleObject(process_, INFINITE);
    }
    int code = 0;
    Exited(&code);
    return code;
}

#else  // POSIX

std::unique_ptr<WorkerProcess> WorkerProcess::Spawn(const std::string& executable,
                                                    const std::vector<std::string>& args, std::string* error) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
        SetError(error, std::string("socketpair failed: ") + std::strerror(errno));
        return nullptr;
    }
    fcntl(pair[0], F_SETFD, FD_CLOEXEC);  // the listener's end never reaches a worker

    // Everything the child needs is built before fork(): after it, a
    // multithreaded parent's child may only make async-signal-safe calls.
    std::vector<std::string> storage;
    storage.push_back(executable);
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& arg : storage) argv.push_back(arg.data());
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        close(pair[0]);
        close(pair[1]);
        SetError(error, std::string("fork failed: ") + std::strerror(errno));
        return nullptr;
    }
    if (pid == 0) {
        // No PR_SET_PDEATHSIG: it fires when the forking *thread* exits, and
        // the listener forks from short-lived routing threads. Workers notice
        // a dead listener through EOF on the channel instead (see RunWorker).
        if (pair[1] == 3) {
            fcntl(3, F_SETFD, 0);
        } else if (dup2(pair[1], 3) < 0) {
            _exit(127);
        }
        execv(argv[0], argv.data());
        _exit(127);
    }
    close(pair[1]);

    std::unique_ptr<WorkerProcess> worker(new WorkerProcess());
    worker->pid_ = static_cast<uint32_t>(pid);
    worker->channel_ = pair[0];
    return worker;
}

WorkerProcess::~WorkerProcess() {
    if (pid_ != 0 && !Exited()) StopAndWait(std::chrono::seconds(1));
    CloseChannel();
}

bool WorkerProcess::HandOff(const HandoffHeader& header, tunnel::TcpStream stream,
                            std::chrono::milliseconds timeout) {
    if (channel_ < 0) return false;
    std::string message = EncodeHeader(header);
    iovec iov{message.data(), message.size()};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    const int fd = static_cast<int>(stream.native());
    std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
    if (sendmsg(channel_, &msg, MSG_NOSIGNAL) != static_cast<ssize_t>(message.size())) return false;
    return ReadAck(timeout);  // the kernel already duplicated the fd; `stream` closes ours
}

bool WorkerProcess::ReadAck(std::chrono::milliseconds timeout) {
    pollfd fd{channel_, POLLIN, 0};
    if (poll(&fd, 1, static_cast<int>(timeout.count())) <= 0) return false;
    char ack = 0;
    return recv(channel_, &ack, 1, 0) == 1 && ack == 1;
}

bool WorkerProcess::Exited(int* code) {
    if (!exit_code_) {
        int status = 0;
        if (waitpid(static_cast<pid_t>(pid_), &status, WNOHANG) == static_cast<pid_t>(pid_)) {
            exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        }
    }
    if (exit_code_ && code) *code = *exit_code_;
    return exit_code_.has_value();
}

void WorkerProcess::CloseChannel() {
    if (channel_ >= 0) {
        shutdown(channel_, SHUT_RDWR);
        close(channel_);
        channel_ = -1;
    }
}

int WorkerProcess::StopAndWait(std::chrono::milliseconds timeout) {
    CloseChannel();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!Exited() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!Exited()) {
        kill(static_cast<pid_t>(pid_), SIGKILL);
        int status = 0;
        waitpid(static_cast<pid_t>(pid_), &status, 0);
        exit_code_ = 128 + SIGKILL;
    }
    return *exit_code_;
}

#endif

// ------------------------------------------------------------ WorkerChannel

#ifdef _WIN32

std::unique_ptr<WorkerChannel> WorkerChannel::FromInherited(const std::string& ack_handle, std::string* error) {
    std::unique_ptr<WorkerChannel> channel(new WorkerChannel());
    channel->from_listener_ = GetStdHandle(STD_INPUT_HANDLE);
    try {
        channel->to_listener_ = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::stoull(ack_handle)));
    } catch (...) {
        SetError(error, "missing or invalid --ack-handle");
        return nullptr;
    }
    if (!channel->from_listener_ || channel->from_listener_ == INVALID_HANDLE_VALUE) {
        SetError(error, "no IPC channel on stdin (workers are started by the listener)");
        return nullptr;
    }
    return channel;
}

WorkerChannel::~WorkerChannel() {
    if (to_listener_) CloseHandle(to_listener_);
}

std::optional<std::pair<HandoffHeader, tunnel::TcpStream>> WorkerChannel::Receive() {
    char message[kHeaderBytes + sizeof(WSAPROTOCOL_INFOW)];
    if (!ReadFull(from_listener_, message, sizeof(message))) return std::nullopt;  // EOF: stop
    WSAPROTOCOL_INFOW info;
    std::memcpy(&info, message + kHeaderBytes, sizeof(info));
    tunnel::InitSockets();
    const SOCKET s = WSASocketW(FROM_PROTOCOL_INFO, FROM_PROTOCOL_INFO, FROM_PROTOCOL_INFO, &info, 0,
                                WSA_FLAG_OVERLAPPED);
    const char ack = s == INVALID_SOCKET ? 0 : 1;
    if (!WriteFull(to_listener_, &ack, 1) || s == INVALID_SOCKET) return std::nullopt;
    return std::make_pair(DecodeHeader(message), tunnel::TcpStream(static_cast<tunnel::NativeSocket>(s)));
}

std::string CurrentExecutablePath() {
    std::wstring path(32768, L'\0');
    const DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    path.resize(size);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0,
                                          nullptr, nullptr);
    std::string utf8(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), utf8.data(), bytes, nullptr,
                        nullptr);
    return utf8;
}

#else

std::unique_ptr<WorkerChannel> WorkerChannel::FromInherited(const std::string&, std::string* error) {
    if (fcntl(3, F_GETFD) < 0) {
        SetError(error, "no IPC channel on fd 3 (workers are started by the listener)");
        return nullptr;
    }
    fcntl(3, F_SETFD, FD_CLOEXEC);
    std::unique_ptr<WorkerChannel> channel(new WorkerChannel());
    channel->channel_ = 3;
    return channel;
}

WorkerChannel::~WorkerChannel() {
    if (channel_ >= 0) close(channel_);
}

std::optional<std::pair<HandoffHeader, tunnel::TcpStream>> WorkerChannel::Receive() {
    char message[kHeaderBytes];
    size_t got = 0;
    int fd = -1;
    while (got < sizeof(message)) {
        iovec iov{message + got, sizeof(message) - got};
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
#ifdef MSG_CMSG_CLOEXEC
        const ssize_t n = recvmsg(channel_, &msg, MSG_CMSG_CLOEXEC);
#else
        const ssize_t n = recvmsg(channel_, &msg, 0);
#endif
        if (n <= 0) {
            if (fd >= 0) close(fd);
            return std::nullopt;  // EOF: the listener asked us to stop
        }
        got += static_cast<size_t>(n);
        for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) std::memcpy(&fd, CMSG_DATA(c), sizeof(int));
        }
    }
    const char ack = fd >= 0 ? 1 : 0;
    if (send(channel_, &ack, 1, MSG_NOSIGNAL) != 1 || fd < 0) {
        if (fd >= 0) close(fd);
        return std::nullopt;
    }
    return std::make_pair(DecodeHeader(message), tunnel::TcpStream(static_cast<tunnel::NativeSocket>(fd)));
}

std::string CurrentExecutablePath() {
    std::string path(4096, '\0');
    const ssize_t size = readlink("/proc/self/exe", path.data(), path.size());
    path.resize(size > 0 ? static_cast<size_t>(size) : 0);
    return path;
}

#endif

}  // namespace tuna
