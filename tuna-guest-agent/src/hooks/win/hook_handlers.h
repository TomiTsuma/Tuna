#pragma once

#include <windows.h>
#include <ntstatus.h>

#ifdef __cplusplus

extern "C" {
    #endif
    NTSTATUS WINAPI HookedNtCreateFile(
        PHANDLE FileHandle,
        ACCESS_MASK DesiredAccess,
        POBJECT_ATTRIBUTES ObjectAttributes,
        PIO_STATUS_BLOCK IoStatusBlock,
        PLARGE_INTEGER AllocationSize,
        ULONG FileAttributes,
        ULONG ShareAccess,
        ULONG CreateDisposition,
        ULONG CreateOptions,
        PVOID EaBuffer,
        ULONG EaLength
    );

    bool handlers_init();

    void handlers_uninit();

    #ifdef __cplusplus
}
#endif