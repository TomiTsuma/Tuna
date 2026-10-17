/* Tracing pass-through nvcuda.dll (Phase 3 spike, kept as a coverage tool).
 *
 * Forwards every driver API call to the real System32\nvcuda.dll and counts
 * it. cuGetProcAddress(_v2) results are swapped for our counting thunks, so
 * calls made through function pointers are counted too. cuGetExportTable
 * requests (the undocumented tables cudart uses) are logged by UUID.
 *
 * At process exit the report is written to %TUNA_TRACE_LOG%, or
 * %TEMP%\tuna_nvcuda_trace.txt.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "names.inc"

typedef int CUresult;
typedef struct { unsigned char bytes[16]; } CUuuid;

/* Filled at load time; read by the generated thunks. */
uint64_t tuna_real[TUNA_EXPORT_COUNT];
volatile uint64_t tuna_counts[TUNA_EXPORT_COUNT];
extern void* const tuna_thunks[TUNA_EXPORT_COUNT];

#define MAX_RECORDS 2048
static CRITICAL_SECTION g_lock;
static struct { char symbol[96]; int version; uint64_t flags; int found; unsigned count; } g_procs[MAX_RECORDS];
static int g_proc_count;
static struct { CUuuid id; unsigned count; CUresult result; const void* table; } g_tables[128];
static int g_table_count;

static int IndexOf(const char* name) {
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i)
        if (strcmp(tuna_names[i], name) == 0) return i;
    return -1;
}

CUresult tuna_cuGetProcAddress(const char*, void**, int, uint64_t);
CUresult tuna_cuGetProcAddress_v2(const char*, void**, int, uint64_t, int*);
CUresult tuna_cuGetExportTable(const void**, const CUuuid*);

/* Replaces a real driver function pointer with what we export for it: the
 * C wrapper for the three wrapped functions (cudart's first act is to look up
 * cuGetProcAddress itself, and every later lookup goes through that pointer),
 * otherwise the counting thunk. */
static void* Wrap(void* real) {
    if (!real) return real;
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i) {
        if ((void*)(uintptr_t)tuna_real[i] != real) continue;
        if (strcmp(tuna_names[i], "cuGetProcAddress") == 0) return (void*)tuna_cuGetProcAddress;
        if (strcmp(tuna_names[i], "cuGetProcAddress_v2") == 0) return (void*)tuna_cuGetProcAddress_v2;
        if (strcmp(tuna_names[i], "cuGetExportTable") == 0) return (void*)tuna_cuGetExportTable;
        return tuna_thunks[i];
    }
    return real; /* not an export (e.g. an internal entry point): pass through */
}

static void RecordProc(const char* symbol, int version, uint64_t flags, int found) {
    EnterCriticalSection(&g_lock);
    for (int i = 0; i < g_proc_count; ++i) {
        if (g_procs[i].version == version && g_procs[i].flags == flags && strcmp(g_procs[i].symbol, symbol) == 0) {
            g_procs[i].count++;
            LeaveCriticalSection(&g_lock);
            return;
        }
    }
    if (g_proc_count < MAX_RECORDS) {
        lstrcpynA(g_procs[g_proc_count].symbol, symbol, sizeof(g_procs[0].symbol));
        g_procs[g_proc_count].version = version;
        g_procs[g_proc_count].flags = flags;
        g_procs[g_proc_count].found = found;
        g_procs[g_proc_count].count = 1;
        g_proc_count++;
    }
    LeaveCriticalSection(&g_lock);
}

typedef CUresult (*GetProcAddressFn)(const char*, void**, int, uint64_t);
typedef CUresult (*GetProcAddressV2Fn)(const char*, void**, int, uint64_t, int*);
typedef CUresult (*GetExportTableFn)(const void**, const CUuuid*);

/* The driver's cuGetProcAddress does not return exported addresses for every
 * symbol (it did not for cuGetProcAddress itself), so the functions we wrap
 * are matched by name. For CUDA >= 12.0 callers "cuGetProcAddress" means the
 * _v2 signature. Returns 1 if *pfn was replaced. */
static int WrapByName(const char* symbol, int version, void** pfn) {
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = version >= 12000 ? (void*)tuna_cuGetProcAddress_v2 : (void*)tuna_cuGetProcAddress;
        return 1;
    }
    if (strcmp(symbol, "cuGetExportTable") == 0) {
        *pfn = (void*)tuna_cuGetExportTable;
        return 1;
    }
    return 0;
}

/* found: 0 = driver has no such symbol, 1 = wrapped/counted, 2 = pointer is not
 * an export (calls through it are not counted). */
static int Resolve(CUresult rc, const char* symbol, int version, void** pfn) {
    if (rc != 0 || !pfn || !*pfn) return 0;
    if (WrapByName(symbol, version, pfn)) return 1;
    void* wrapped = Wrap(*pfn);
    const int found = wrapped != *pfn ? 1 : 2;
    *pfn = wrapped;
    return found;
}

__declspec(dllexport) CUresult tuna_cuGetProcAddress(const char* symbol, void** pfn, int version, uint64_t flags) {
    const CUresult rc = ((GetProcAddressFn)(uintptr_t)tuna_real[IndexOf("cuGetProcAddress")])(symbol, pfn, version, flags);
    RecordProc(symbol, version, flags, Resolve(rc, symbol, version, pfn));
    return rc;
}

__declspec(dllexport) CUresult tuna_cuGetProcAddress_v2(const char* symbol, void** pfn, int version, uint64_t flags,
                                                       int* status) {
    const CUresult rc =
        ((GetProcAddressV2Fn)(uintptr_t)tuna_real[IndexOf("cuGetProcAddress_v2")])(symbol, pfn, version, flags, status);
    RecordProc(symbol, version, flags, Resolve(rc, symbol, version, pfn));
    return rc;
}

__declspec(dllexport) CUresult tuna_cuGetExportTable(const void** table, const CUuuid* id) {
    const CUresult rc = ((GetExportTableFn)(uintptr_t)tuna_real[IndexOf("cuGetExportTable")])(table, id);
    EnterCriticalSection(&g_lock);
    int i = 0;
    while (i < g_table_count && memcmp(&g_tables[i].id, id, sizeof(CUuuid)) != 0) ++i;
    if (i == g_table_count && g_table_count < 128) {
        g_tables[i].id = *id;
        g_tables[i].result = rc;
        g_tables[i].table = (rc == 0 && table) ? *table : NULL;
        g_table_count++;
    }
    /* The driver's table is returned untouched: these tables are internal to
     * NVIDIA's runtime/driver pair and are integrity-checked, so the tracer
     * only records which ones are requested. */
    if (i < g_table_count) g_tables[i].count++;
    LeaveCriticalSection(&g_lock);
    return rc;
}

static void WriteReport(void) {
    char path[MAX_PATH];
    if (!GetEnvironmentVariableA("TUNA_TRACE_LOG", path, sizeof(path))) {
        GetTempPathA(sizeof(path), path);
        lstrcatA(path, "tuna_nvcuda_trace.txt");
    }
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return;
    char line[256];
    DWORD written;
#define EMIT(...) do { int n = wsprintfA(line, __VA_ARGS__); WriteFile(file, line, (DWORD)n, &written, NULL); } while (0)
    EMIT("# calls per export (count name)\r\n");
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i)
        if (tuna_counts[i]) EMIT("call %lu %s\r\n", (unsigned long)tuna_counts[i], tuna_names[i]);
    EMIT("# cuGetProcAddress requests (count symbol version flags found)\r\n");
    for (int i = 0; i < g_proc_count; ++i)
        EMIT("proc %u %s %d %lu %d\r\n", g_procs[i].count, g_procs[i].symbol, g_procs[i].version,
             (unsigned long)g_procs[i].flags, g_procs[i].found);
    EMIT("# cuGetExportTable requests (count uuid result)\r\n");
    for (int i = 0; i < g_table_count; ++i) {
        char hex[33];
        for (int b = 0; b < 16; ++b) wsprintfA(hex + 2 * b, "%02x", g_tables[i].id.bytes[b]);
        EMIT("table %u %s %d\r\n", g_tables[i].count, hex, g_tables[i].result);
    }
#undef EMIT
    CloseHandle(file);
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    if (reason == DLL_PROCESS_ATTACH) {
        InitializeCriticalSection(&g_lock);
        wchar_t path[MAX_PATH];
        const UINT n = GetSystemDirectoryW(path, MAX_PATH);
        lstrcpyW(path + n, L"\\nvcuda.dll");
        /* A full path loads the real driver even though a module with the same
         * base name (this one) is already loaded. */
        HMODULE real = LoadLibraryW(path);
        if (!real) return FALSE;
        for (int i = 0; i < TUNA_EXPORT_COUNT; ++i)
            tuna_real[i] = (uint64_t)(uintptr_t)GetProcAddress(real, tuna_names[i]);
    } else if (reason == DLL_PROCESS_DETACH) {
        (void)reserved;
        WriteReport();
    }
    return TRUE;
}
