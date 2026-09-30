/* membw: measures sustained RAM read bandwidth (GB/s). One small C file, no dependencies.
 * usage: membw <threads> <mib_per_thread> [passes]
 * Each thread reads its own buffer after touching it itself, so on NUMA machines the memory is local to the core it is
 * pinned to (Linux). Prints one line: "membw_gbs <number>". Built by CI for releases, or by the installer with cc. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <sched.h>
#endif

static size_t bytes; static int passes;
static volatile uint64_t sink;

/* a small portable barrier (macOS has no pthread_barrier) */
typedef struct { pthread_mutex_t m; pthread_cond_t c; int n, waiting, phase; } barrier_t;
static barrier_t barrier;
static void barrier_init(barrier_t *b, int n) { pthread_mutex_init(&b->m, NULL); pthread_cond_init(&b->c, NULL); b->n = n; b->waiting = 0; b->phase = 0; }
static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->m); int ph = b->phase;
    if (++b->waiting == b->n) { b->waiting = 0; b->phase++; pthread_cond_broadcast(&b->c); }
    else while (ph == b->phase) pthread_cond_wait(&b->c, &b->m);
    pthread_mutex_unlock(&b->m);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *worker(void *arg) {
    long id = (long)arg;
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set); CPU_SET((int)id, &set); sched_setaffinity(0, sizeof set, &set);
#endif
    uint64_t *buf = aligned_alloc(64, bytes);
    if (!buf) return NULL;
    memset(buf, 1, bytes);                              /* first touch by the pinned thread */
    size_t n = bytes / sizeof(uint64_t); uint64_t acc = 0;
    barrier_wait(&barrier);
    for (int p = 0; p < passes; p++)
        for (size_t i = 0; i < n; i += 4) acc += buf[i] + buf[i + 1] + buf[i + 2] + buf[i + 3];
    sink = acc; free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: membw <threads> <mib_per_thread> [passes]\n"); return 2; }
    int threads = atoi(argv[1]); bytes = (size_t)atol(argv[2]) << 20; passes = argc > 3 ? atoi(argv[3]) : 4;
    if (threads < 1 || threads > 512 || bytes < (1u << 20)) return 2;
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
        pthread_t th[512]; barrier_init(&barrier, threads + 1);
        for (long i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void *)i);
        barrier_wait(&barrier); double t0 = now();
        for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
        double dt = now() - t0, gbs = (double)bytes * passes * threads / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    printf("membw_gbs %.2f\n", best);
    return 0;
}
