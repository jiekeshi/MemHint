/**
 * Hard Case 4: State Machine with Multi-phase Lifecycle
 *
 * 对象有多个生命周期阶段，不同阶段需要不同的清理逻辑。
 * 常见于网络连接、文件句柄、数据库事务等。
 *
 * 难点：
 * - 状态转换的正确性
 * - 每个状态可能有不同的资源需要清理
 * - 异常状态下的资源回收
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ========== 自定义分配器 ==========

void* session_alloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

void session_free(void* ptr) {
    free(ptr);
}

char* session_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* dup = (char*)session_alloc(len);
    if (dup) memcpy(dup, s, len);
    return dup;
}

// ========== 状态定义 ==========

typedef enum {
    STATE_INIT = 0,
    STATE_CONNECTING,
    STATE_AUTHENTICATING,
    STATE_READY,
    STATE_BUSY,
    STATE_CLOSING,
    STATE_CLOSED,
    STATE_ERROR
} SessionState;

const char* state_name(SessionState s) {
    switch(s) {
        case STATE_INIT: return "INIT";
        case STATE_CONNECTING: return "CONNECTING";
        case STATE_AUTHENTICATING: return "AUTHENTICATING";
        case STATE_READY: return "READY";
        case STATE_BUSY: return "BUSY";
        case STATE_CLOSING: return "CLOSING";
        case STATE_CLOSED: return "CLOSED";
        case STATE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ========== 会话结构 ==========

typedef struct {
    // 基础信息（INIT 阶段分配）
    char* session_id;
    char* host;
    int port;

    // 连接资源（CONNECTING 阶段分配）
    int socket_fd;
    char* recv_buffer;
    size_t recv_buffer_size;
    char* send_buffer;
    size_t send_buffer_size;

    // 认证信息（AUTHENTICATING 阶段分配）
    char* username;
    char* auth_token;

    // 工作数据（READY/BUSY 阶段分配）
    void* work_context;
    char* last_query;
    char* last_result;

    SessionState state;
    int error_code;
} Session;

// ========== 生命周期函数 ==========

Session* session_create(const char* host, int port) {
    Session* sess = (Session*)session_alloc(sizeof(Session));
    if (!sess) return NULL;

    sess->session_id = session_strdup("sess_12345");
    sess->host = session_strdup(host);
    sess->port = port;
    sess->state = STATE_INIT;
    sess->socket_fd = -1;

    printf("[Session %s] Created in state %s\n", sess->session_id, state_name(sess->state));
    return sess;
}

int session_connect(Session* sess) {
    if (sess->state != STATE_INIT) {
        return -1;
    }

    sess->state = STATE_CONNECTING;
    printf("[Session %s] State -> %s\n", sess->session_id, state_name(sess->state));

    // 分配连接资源
    sess->recv_buffer_size = 4096;
    sess->recv_buffer = (char*)session_alloc(sess->recv_buffer_size);
    if (!sess->recv_buffer) {
        sess->state = STATE_ERROR;
        sess->error_code = -1;
        return -1;
    }

    sess->send_buffer_size = 4096;
    sess->send_buffer = (char*)session_alloc(sess->send_buffer_size);
    if (!sess->send_buffer) {
        // BUG: 分配失败时忘记释放 recv_buffer!
        // session_free(sess->recv_buffer);  // 应该有这行!
        sess->state = STATE_ERROR;
        sess->error_code = -2;
        return -1;  // MEMORY LEAK: recv_buffer 泄漏!
    }

    // 模拟连接成功
    sess->socket_fd = 42;
    return 0;
}

int session_authenticate(Session* sess, const char* username, const char* password) {
    if (sess->state != STATE_CONNECTING) {
        return -1;
    }

    sess->state = STATE_AUTHENTICATING;
    printf("[Session %s] State -> %s\n", sess->session_id, state_name(sess->state));

    sess->username = session_strdup(username);
    if (!sess->username) {
        sess->state = STATE_ERROR;
        return -1;
    }

    // 模拟认证过程
    if (strcmp(password, "secret") == 0) {
        sess->auth_token = session_strdup("token_abc123");
        sess->state = STATE_READY;
        printf("[Session %s] State -> %s\n", sess->session_id, state_name(sess->state));
        return 0;
    } else {
        // BUG: 认证失败时没有清理 username!
        // session_free(sess->username);  // 应该有这行!
        sess->state = STATE_ERROR;
        sess->error_code = -3;
        return -1;  // MEMORY LEAK: username 泄漏!
    }
}

int session_execute(Session* sess, const char* query) {
    if (sess->state != STATE_READY) {
        return -1;
    }

    sess->state = STATE_BUSY;
    printf("[Session %s] State -> %s, executing: %s\n",
           sess->session_id, state_name(sess->state), query);

    // 释放上次的查询/结果
    if (sess->last_query) session_free(sess->last_query);
    if (sess->last_result) session_free(sess->last_result);

    sess->last_query = session_strdup(query);
    sess->last_result = session_strdup("Result: OK");

    sess->state = STATE_READY;
    return 0;
}

void session_close(Session* sess) {
    if (sess->state == STATE_CLOSED) {
        return;
    }

    sess->state = STATE_CLOSING;
    printf("[Session %s] State -> %s\n", sess->session_id, state_name(sess->state));

    // 清理工作数据
    if (sess->last_query) {
        session_free(sess->last_query);
        sess->last_query = NULL;
    }
    if (sess->last_result) {
        session_free(sess->last_result);
        sess->last_result = NULL;
    }
    if (sess->work_context) {
        session_free(sess->work_context);
        sess->work_context = NULL;
    }

    // 清理认证信息
    if (sess->auth_token) {
        session_free(sess->auth_token);
        sess->auth_token = NULL;
    }
    // BUG: 忘记清理 username!
    // session_free(sess->username);  // 应该有这行!

    // 清理连接资源
    if (sess->recv_buffer) {
        session_free(sess->recv_buffer);
        sess->recv_buffer = NULL;
    }
    if (sess->send_buffer) {
        session_free(sess->send_buffer);
        sess->send_buffer = NULL;
    }
    sess->socket_fd = -1;

    sess->state = STATE_CLOSED;
    printf("[Session %s] State -> %s\n", sess->session_id, state_name(sess->state));
}

void session_destroy(Session* sess) {
    if (!sess) return;

    // 如果还没关闭，先关闭
    if (sess->state != STATE_CLOSED) {
        session_close(sess);
    }

    // 清理基础信息
    if (sess->session_id) session_free(sess->session_id);
    if (sess->host) session_free(sess->host);

    // BUG: 如果 session_close 没有清理某些资源，这里也不会清理
    // 因为 session_close 里漏掉了 username，这里也不会释放它!
    // MEMORY LEAK: username 泄漏!

    session_free(sess);
    printf("[Session] Destroyed\n");
}

// ========== 错误状态处理（特别容易出 bug）==========

void session_force_close_on_error(Session* sess) {
    printf("[Session %s] Force closing due to error %d\n",
           sess->session_id, sess->error_code);

    // BUG: 错误状态下的清理不完整!
    // 只释放了部分资源

    if (sess->socket_fd >= 0) {
        sess->socket_fd = -1;
    }

    // 漏掉了:
    // - recv_buffer
    // - send_buffer
    // - username
    // - auth_token
    // 等等...

    sess->state = STATE_CLOSED;
    // MEMORY LEAK: 大量资源泄漏!
}

// ========== 测试 ==========

void test_normal_lifecycle() {
    printf("\n=== Test: Normal Lifecycle ===\n");

    Session* sess = session_create("localhost", 5432);
    session_connect(sess);
    session_authenticate(sess, "admin", "secret");
    session_execute(sess, "SELECT * FROM users");
    session_execute(sess, "SELECT * FROM orders");
    session_close(sess);
    session_destroy(sess);
}

void test_connect_failure() {
    printf("\n=== Test: Connect Failure ===\n");

    Session* sess = session_create("localhost", 5432);

    // 模拟 send_buffer 分配失败的场景
    // (在实际代码中，我们无法直接触发，这里只是说明问题)

    session_connect(sess);  // 假设这里 send_buffer 分配失败

    // 如果连接失败，应该如何清理？
    session_destroy(sess);  // recv_buffer 可能泄漏!
}

void test_auth_failure() {
    printf("\n=== Test: Auth Failure ===\n");

    Session* sess = session_create("localhost", 5432);
    session_connect(sess);

    // 认证失败
    int result = session_authenticate(sess, "admin", "wrong_password");
    printf("Auth result: %d (expected -1)\n", result);

    // 销毁会话
    session_destroy(sess);  // username 泄漏!
}

void test_error_force_close() {
    printf("\n=== Test: Error Force Close ===\n");

    Session* sess = session_create("localhost", 5432);
    session_connect(sess);
    session_authenticate(sess, "admin", "secret");

    // 模拟运行时错误
    sess->state = STATE_ERROR;
    sess->error_code = -99;

    session_force_close_on_error(sess);  // 大量资源泄漏!
    session_destroy(sess);
}

int main() {
    test_normal_lifecycle();
    test_connect_failure();
    test_auth_failure();
    test_error_force_close();
    return 0;
}