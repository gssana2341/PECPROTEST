// audit_log.c — audit logging สำหรับ production
//
// Log ทุก operation สำคัญ สำหรับ debugging, compliance, และ security audit
// รองรับหลาย log level และ output target

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdarg.h>

// ─── Log Levels ─────────────────────────────────────────────────────

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_CRITICAL
} LogLevel;

// ─── Audit Logger State ─────────────────────────────────────────────

static FILE     *g_log_file    = NULL;
static LogLevel  g_min_level   = LOG_INFO;

static const char *level_names[] = {
    "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"
};

// Initialize audit logging
int audit_init(const char *log_path, LogLevel min_level) {
    g_min_level = min_level;
    if (log_path) {
        g_log_file = fopen(log_path, "a");
        if (!g_log_file) return -1;
    } else {
        g_log_file = stderr;
    }
    return 0;
}

// Log an audit entry
void audit_log(LogLevel level, const char *op, const char *detail) {
    if (level < g_min_level || !g_log_file) return;

    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", t);

    fprintf(g_log_file, "[%s] [%s] %-20s %s\n",
            timestamp, level_names[level], op, detail ? detail : "");
    fflush(g_log_file);
}

// Shutdown logging
void audit_shutdown(void) {
    if (g_log_file && g_log_file != stderr) {
        fclose(g_log_file);
    }
    g_log_file = NULL;
}
