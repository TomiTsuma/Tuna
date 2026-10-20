/* Tracing pass-through libcudnn.so.9 (Phase 3g coverage tool, Linux).
 *
 * Forwards every cuDNN export to the real library ($TUNA_REAL_CUDNN) and
 * counts it. The backend (graph) API is wrapped so the report also shows
 * which descriptor types are created and finalized, and which attributes are
 * set and read (by descriptor type, attribute name and element type).
 *
 * At process exit the report is written to $TUNA_TRACE_LOG, or stderr.
 * Preload it ahead of PyTorch's own copy (PyTorch finds cuDNN by RPATH):
 *   LD_PRELOAD=bin/libcudnn.so.9 TUNA_REAL_CUDNN=/path/to/real/libcudnn.so.9 python app.py
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "names.inc"

/* Filled at load time; read by the generated thunks. */
uint64_t tuna_real[TUNA_EXPORT_COUNT];
volatile uint64_t tuna_counts[TUNA_EXPORT_COUNT];

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* Descriptor pointer -> backend descriptor type (open addressing). */
#define DESC_SLOTS 65536
static struct { void* desc; int type; } g_descs[DESC_SLOTS];

static size_t Slot(void* desc) { return ((uintptr_t)desc >> 4) * 2654435761u % DESC_SLOTS; }

static void RememberDesc(void* desc, int type) {
    for (size_t i = Slot(desc), n = 0; n < DESC_SLOTS; i = (i + 1) % DESC_SLOTS, ++n) {
        if (g_descs[i].desc == NULL || g_descs[i].desc == desc) {
            g_descs[i].desc = desc;
            g_descs[i].type = type;
            return;
        }
    }
}

static int DescType(void* desc) {
    for (size_t i = Slot(desc), n = 0; n < DESC_SLOTS && g_descs[i].desc; i = (i + 1) % DESC_SLOTS, ++n) {
        if (g_descs[i].desc == desc) return g_descs[i].type;
    }
    return -1;
}

/* Counters keyed by up to three small ints. */
#define MAX_RECORDS 4096
typedef struct { char kind; int a, b, c; uint64_t count, max_elements; } Record;
static Record g_records[MAX_RECORDS];
static int g_record_count;

static void Count(char kind, int a, int b, int c, int64_t elements) {
    pthread_mutex_lock(&g_lock);
    Record* r = NULL;
    for (int i = 0; i < g_record_count; ++i) {
        Record* x = &g_records[i];
        if (x->kind == kind && x->a == a && x->b == b && x->c == c) {
            r = x;
            break;
        }
    }
    if (!r && g_record_count < MAX_RECORDS) {
        r = &g_records[g_record_count++];
        r->kind = kind;
        r->a = a;
        r->b = b;
        r->c = c;
    }
    if (r) {
        r->count++;
        if (elements > 0 && (uint64_t)elements > r->max_elements) r->max_elements = (uint64_t)elements;
    }
    pthread_mutex_unlock(&g_lock);
}

static int IndexOf(const char* name) {
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i)
        if (strcmp(tuna_names[i], name) == 0) return i;
    return -1;
}

#define REAL(name) ((__typeof__(&name))(uintptr_t)tuna_real[TUNA_INDEX_##name])
#define BUMP(name) __atomic_add_fetch(&tuna_counts[TUNA_INDEX_##name], 1, __ATOMIC_RELAXED)

/* The backend API, declared with plain ints for its enums. */
int cudnnBackendCreateDescriptor(int type, void** desc);
int cudnnBackendDestroyDescriptor(void* desc);
int cudnnBackendSetAttribute(void* desc, int name, int type, int64_t count, const void* elements);
int cudnnBackendGetAttribute(void* desc, int name, int type, int64_t requested, int64_t* count, void* elements);
int cudnnBackendFinalize(void* desc);
int cudnnBackendExecute(void* handle, void* plan, void* variant_pack);

int cudnnBackendCreateDescriptor(int type, void** desc) {
    BUMP(cudnnBackendCreateDescriptor);
    const int status = REAL(cudnnBackendCreateDescriptor)(type, desc);
    Count('C', type, 0, status, 0);
    if (status == 0 && desc) {
        pthread_mutex_lock(&g_lock);
        RememberDesc(*desc, type);
        pthread_mutex_unlock(&g_lock);
    }
    return status;
}

int cudnnBackendDestroyDescriptor(void* desc) {
    BUMP(cudnnBackendDestroyDescriptor);
    return REAL(cudnnBackendDestroyDescriptor)(desc);
}

int cudnnBackendSetAttribute(void* desc, int name, int type, int64_t count, const void* elements) {
    BUMP(cudnnBackendSetAttribute);
    const int status = REAL(cudnnBackendSetAttribute)(desc, name, type, count, elements);
    pthread_mutex_lock(&g_lock);
    const int desc_type = DescType(desc);
    pthread_mutex_unlock(&g_lock);
    Count('S', desc_type, name, type * 10000 + status, count);
    return status;
}

int cudnnBackendGetAttribute(void* desc, int name, int type, int64_t requested, int64_t* count, void* elements) {
    BUMP(cudnnBackendGetAttribute);
    const int status = REAL(cudnnBackendGetAttribute)(desc, name, type, requested, count, elements);
    pthread_mutex_lock(&g_lock);
    const int desc_type = DescType(desc);
    pthread_mutex_unlock(&g_lock);
    Count('G', desc_type, name, type * 10000 + status, elements ? requested : -1);
    return status;
}

int cudnnBackendFinalize(void* desc) {
    BUMP(cudnnBackendFinalize);
    const int status = REAL(cudnnBackendFinalize)(desc);
    pthread_mutex_lock(&g_lock);
    const int desc_type = DescType(desc);
    pthread_mutex_unlock(&g_lock);
    Count('F', desc_type, 0, status, 0);
    return status;
}

int cudnnBackendExecute(void* handle, void* plan, void* variant_pack) {
    BUMP(cudnnBackendExecute);
    const int status = REAL(cudnnBackendExecute)(handle, plan, variant_pack);
    Count('E', 0, 0, status, 0);
    return status;
}

__attribute__((constructor)) static void Load(void) {
    const char* path = getenv("TUNA_REAL_CUDNN");
    if (!path || !*path) {
        fprintf(stderr, "cudnn_trace: set TUNA_REAL_CUDNN to the real libcudnn.so.9\n");
        abort();
    }
    void* real = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!real) {
        fprintf(stderr, "cudnn_trace: %s\n", dlerror());
        abort();
    }
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i) {
        tuna_real[i] = (uint64_t)(uintptr_t)dlsym(real, tuna_names[i]);
    }
    (void)IndexOf;
}

__attribute__((destructor)) static void Report(void) {
    const char* path = getenv("TUNA_TRACE_LOG");
    FILE* f = path && *path ? fopen(path, "w") : stderr;
    if (!f) return;
    fprintf(f, "== calls per export\n");
    for (int i = 0; i < TUNA_EXPORT_COUNT; ++i) {
        if (tuna_counts[i]) fprintf(f, "%10llu %s\n", (unsigned long long)tuna_counts[i], tuna_names[i]);
    }
    static const char* const kinds = "CFSGE";
    static const char* const titles[] = {
        "create: descriptor type, status",
        "finalize: descriptor type, status",
        "set: descriptor type, attribute, element type, status (max elements)",
        "get: descriptor type, attribute, element type, status (max requested; -1 = count query)",
        "execute: status",
    };
    for (int k = 0; k < 5; ++k) {
        fprintf(f, "== %s\n", titles[k]);
        for (int i = 0; i < g_record_count; ++i) {
            const Record* r = &g_records[i];
            if (r->kind != kinds[k]) continue;
            if (r->kind == 'S' || r->kind == 'G') {
                fprintf(f, "%10llu desc=%d attr=%d type=%d status=%d max=%lld\n", (unsigned long long)r->count, r->a,
                        r->b, r->c / 10000, r->c % 10000, (long long)r->max_elements);
            } else {
                fprintf(f, "%10llu desc=%d status=%d\n", (unsigned long long)r->count, r->a, r->c);
            }
        }
    }
    if (f != stderr) fclose(f);
}
