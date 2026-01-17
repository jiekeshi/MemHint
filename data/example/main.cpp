#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

// ============== 基础数据结构 ==============

typedef struct listNode {
    struct listNode *prev;
    struct listNode *next;
    void *value;
} listNode;

typedef struct list {
    listNode *head;
    listNode *tail;
    unsigned long len;
} list;

typedef struct listIter {
    listNode *next;
    int direction;
} listIter;

typedef struct dict {
    void **table;
    unsigned long size;
    unsigned long used;
} dict;

typedef struct buffer {
    char *data;
    size_t len;
    size_t cap;
} buffer;

typedef struct connection {
    int fd;
    int state;
    buffer *input;
    buffer *output;
    struct client *owner;
} connection;

typedef struct client {
    int id;
    int replstate;
    int flags;
    time_t repl_ack_time;
    time_t repl_last_partial_write;
    char name[32];
    connection *conn;
    list *pending_commands;
    dict *pubsub_channels;
    struct client *master;
    list *slaves;
    buffer *querybuf;
    int refcount;
    int authenticated;
    void *privdata;
} client;

typedef struct command {
    char *name;
    int argc;
    char **argv;
    client *client;
    int flags;
} command;

typedef void (*commandHandler)(client *c, command *cmd);

typedef struct commandEntry {
    char *name;
    commandHandler handler;
    int flags;
} commandEntry;

// ============== 全局状态 ==============

#define AL_START_HEAD 0
#define CLIENT_PRE_PSYNC (1<<0)
#define CLIENT_BLOCKED (1<<1)
#define CLIENT_CLOSE_ASAP (1<<2)
#define CLIENT_MASTER (1<<3)
#define CLIENT_SLAVE (1<<4)
#define CLIENT_PENDING_WRITE (1<<5)
#define CLIENT_FREED (1<<6)

#define SLAVE_STATE_ONLINE 1
#define SLAVE_STATE_WAIT_BGSAVE_END 2
#define SLAVE_STATE_SEND_BULK 3

#define RDB_CHILD_TYPE_SOCKET 1
#define RDB_CHILD_TYPE_DISK 2

#define CMD_WRITE (1<<0)
#define CMD_READONLY (1<<1)
#define CMD_ADMIN (1<<2)

#define LL_WARNING 1
#define LL_DEBUG 2

struct {
    list *slaves;
    list *clients;
    list *clients_to_close;
    list *clients_pending_write;
    dict *commands;
    time_t unixtime;
    int repl_timeout;
    int rdb_child_type;
    int shutdown_asap;
    client *current_client;
    client *master;
    int loading;
    int max_clients;
} server;

// ============== 链表操作 ==============

list *listCreate(void) {
    list *l = (list *)malloc(sizeof(*l));
    l->head = l->tail = NULL;
    l->len = 0;
    return l;
}

list *listAddNodeTail(list *l, void *value) {
    listNode *node = (listNode *)malloc(sizeof(*node));
    node->value = value;
    node->next = NULL;
    if (l->tail) {
        node->prev = l->tail;
        l->tail->next = node;
        l->tail = node;
    } else {
        node->prev = NULL;
        l->head = l->tail = node;
    }
    l->len++;
    return l;
}

void listDelNode(list *l, listNode *node) {
    if (node->prev)
        node->prev->next = node->next;
    else
        l->head = node->next;
    if (node->next)
        node->next->prev = node->prev;
    else
        l->tail = node->prev;
    free(node);
    l->len--;
}

void listRewind(list *l, listIter *li) {
    li->next = l->head;
    li->direction = AL_START_HEAD;
}

listNode *listNext(listIter *li) {
    listNode *current = li->next;
    if (current != NULL) {
        li->next = current->next;
    }
    return current;
}

listNode *listSearchKey(list *l, void *key) {
    listIter li;
    listNode *ln;
    listRewind(l, &li);
    while ((ln = listNext(&li))) {
        if (ln->value == key) return ln;
    }
    return NULL;
}

// ============== 缓冲区操作 ==============

buffer *bufferCreate(size_t cap) {
    buffer *buf = (buffer *)malloc(sizeof(*buf));
    buf->data = (char *)malloc(cap);
    buf->len = 0;
    buf->cap = cap;
    return buf;
}

void bufferFree(buffer *buf) {
    if (buf) {
        free(buf->data);
        free(buf);
    }
}

void bufferAppend(buffer *buf, const char *data, size_t len) {
    if (buf->len + len > buf->cap) {
        buf->cap = (buf->len + len) * 2;
        buf->data = (char *)realloc(buf->data, buf->cap);
    }
    memcpy(buf->data + buf->len, data, len);
    buf->len += len;
}

// ============== 字典操作 ==============

dict *dictCreate(void) {
    dict *d = (dict *)malloc(sizeof(*d));
    d->size = 16;
    d->used = 0;
    d->table = (void **)calloc(d->size, sizeof(void *));
    return d;
}

void dictFree(dict *d) {
    if (d) {
        free(d->table);
        free(d);
    }
}

// ============== 连接操作 ==============

connection *connCreate(int fd) {
    connection *conn = (connection *)malloc(sizeof(*conn));
    conn->fd = fd;
    conn->state = 0;
    conn->input = bufferCreate(1024);
    conn->output = bufferCreate(1024);
    conn->owner = NULL;
    return conn;
}

void connFree(connection *conn) {
    if (conn) {
        bufferFree(conn->input);
        bufferFree(conn->output);
        free(conn);
    }
}

// ============== 日志函数 ==============

void serverLog(int level, const char *fmt, ...) {
    (void)level;
    va_list args;
    va_start(args, fmt);
    printf("[LOG] ");
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

// ============== Client 操作 ==============

const char *getClientName(client *c) {
    return c ? c->name : "unknown";
}

void unlinkClient(client *c);
void freeClientAsync(client *c);

// 【UAF Pattern 1】: 回调函数中释放后，调用者继续使用
typedef void (*clientCallback)(client *c, void *privdata);

void processClientCallback(client *c, clientCallback cb, void *privdata) {
    // 保存原始状态用于后续检查
    int original_flags = c->flags;

    // 执行回调，回调可能释放 client
    cb(c, privdata);

    // 【UAF】回调可能已经 free 了 c，但这里继续使用
    if (original_flags != c->flags) {
        serverLog(LL_DEBUG, "Client %s flags changed", c->name);
    }
}

// 【UAF Pattern 2】: 引用计数错误
void decrRefCount(client *c) {
    c->refcount--;
    if (c->refcount <= 0) {
        serverLog(LL_DEBUG, "Freeing client %s due to refcount", c->name);
        // 从全局列表移除
        listNode *ln = listSearchKey(server.clients, c);
        if (ln) listDelNode(server.clients, ln);

        connFree(c->conn);
        bufferFree(c->querybuf);
        dictFree(c->pubsub_channels);
        memset(c, 0xDD, sizeof(*c));
        free(c);
    }
}

void incrRefCount(client *c) {
    c->refcount++;
}

// 【UAF Pattern 3】: 异步释放队列，但在加入队列后继续使用
void freeClientAsync(client *c) {
    if (c->flags & CLIENT_FREED) return;
    c->flags |= CLIENT_CLOSE_ASAP | CLIENT_FREED;
    listAddNodeTail(server.clients_to_close, c);
}

void freeClient(client *c) {
    serverLog(LL_DEBUG, "freeClient called for %s", c->name);

    // 从各种列表中移除
    listNode *ln;

    if ((ln = listSearchKey(server.clients, c))) {
        listDelNode(server.clients, ln);
    }

    if ((ln = listSearchKey(server.slaves, c))) {
        listDelNode(server.slaves, ln);
    }

    if ((ln = listSearchKey(server.clients_pending_write, c))) {
        listDelNode(server.clients_pending_write, ln);
    }

    // 如果有 master，通知 master
    if (c->master) {
        c->master->slaves = NULL;  // 简化处理
    }

    // 释放子资源
    connFree(c->conn);
    bufferFree(c->querybuf);
    dictFree(c->pubsub_channels);

    // 释放 pending commands
    if (c->pending_commands) {
        listIter li;
        listRewind(c->pending_commands, &li);
        while ((ln = listNext(&li))) {
            command *cmd = (command *)ln->value;
            free(cmd->name);
            free(cmd);
        }
        free(c->pending_commands);
    }

    memset(c, 0xDD, sizeof(*c));
    free(c);
}

void unlinkClient(client *c) {
    listNode *ln;

    if ((ln = listSearchKey(server.clients, c))) {
        listDelNode(server.clients, ln);
    }

    // 标记为已 unlink，但不释放
    c->conn = NULL;
}

// ============== 复杂的 UAF 场景 ==============

// 【UAF Pattern 4】: 嵌套循环中释放外层迭代变量
void processNestedClients(void) {
    listIter li_outer, li_inner;
    listNode *ln_outer, *ln_inner;

    listRewind(server.clients, &li_outer);
    while ((ln_outer = listNext(&li_outer))) {
        client *c = (client *)ln_outer->value;

        // 内层循环检查是否需要断开此 client
        listRewind(server.slaves, &li_inner);
        while ((ln_inner = listNext(&li_inner))) {
            client *slave = (client *)ln_inner->value;

            // 某些条件下释放外层的 client
            if (slave->master == c && (slave->flags & CLIENT_CLOSE_ASAP)) {
                serverLog(LL_WARNING, "Freeing master %s of closing slave", c->name);
                freeClient(c);
                // 【UAF】break 后，外层循环继续，访问已释放的 c
                break;
            }
        }

        // 【UAF】如果上面 break 了，c 已经被释放
        if (c->flags & CLIENT_PENDING_WRITE) {
            serverLog(LL_DEBUG, "Client %s has pending write", c->name);
        }
    }
}

// 【UAF Pattern 5】: 条件分支中多处释放，后续统一使用
void handleClientState(client *c) {
    int need_log = 0;

    switch (c->replstate) {
        case SLAVE_STATE_ONLINE:
            if ((server.unixtime - c->repl_ack_time) > server.repl_timeout) {
                serverLog(LL_WARNING, "Slave %s timeout in ONLINE state", c->name);
                freeClient(c);
                need_log = 1;
            }
            break;

        case SLAVE_STATE_WAIT_BGSAVE_END:
            if (c->repl_last_partial_write != 0 &&
                (server.unixtime - c->repl_last_partial_write) > server.repl_timeout) {
                serverLog(LL_WARNING, "Slave %s timeout in WAIT_BGSAVE", c->name);
                freeClient(c);
                need_log = 1;
            }
            break;

        case SLAVE_STATE_SEND_BULK:
            if (!c->authenticated) {
                serverLog(LL_WARNING, "Unauthenticated slave %s in SEND_BULK", c->name);
                freeClient(c);
                need_log = 1;
            }
            break;
    }

    // 【UAF】任何一个 case 触发 freeClient 后，这里都会 UAF
    if (need_log) {
        serverLog(LL_DEBUG, "Handled state for client %s, replstate=%d",
                  c->name, c->replstate);
    }
}

// 【UAF Pattern 6】: 通过别名释放
void handleClientWithAlias(client *c) {
    client *target = c;

    // 一些逻辑可能改变 target
    if (c->master) {
        target = c->master;
    }

    // 通过别名释放
    if (target->flags & CLIENT_CLOSE_ASAP) {
        freeClient(target);
    }

    // 【UAF】如果 target == c，则 c 已被释放
    serverLog(LL_DEBUG, "Processed client %s", c->name);
}

// 【UAF Pattern 7】: 跨函数释放 - 被调函数释放调用者的变量
void maybeKillClient(client *c, int aggressive) {
    if (aggressive || (c->flags & CLIENT_CLOSE_ASAP)) {
        freeClient(c);
    }
}

void processClientWithHelper(client *c) {
    // 检查并可能释放
    maybeKillClient(c, c->flags & CLIENT_BLOCKED);

    // 【UAF】不知道 maybeKillClient 是否释放了 c
    if (c->querybuf && c->querybuf->len > 0) {
        serverLog(LL_DEBUG, "Client %s has %zu bytes in querybuf",
                  c->name, c->querybuf->len);
    }
}

// 【UAF Pattern 8】: 事件处理中释放
typedef struct event {
    int type;
    void *data;
    client *client;
} event;

void processEvent(event *ev) {
    client *c = ev->client;

    switch (ev->type) {
        case 0:  // disconnect event
            freeClient(c);
            break;
        case 1:  // data event
            bufferAppend(c->querybuf, (char *)ev->data, strlen((char *)ev->data));
            break;
        case 2:  // error event
            if (c->flags & CLIENT_MASTER) {
                // 释放 master 时特殊处理
                server.master = NULL;
                freeClient(c);
            }
            break;
    }

    // 【UAF】type 0 或 type 2 (if master) 会释放 c
    serverLog(LL_DEBUG, "Processed event type %d for client %s", ev->type, c->name);
}

// 【UAF Pattern 9】: 迭代器失效 + 回调
void forEachClient(clientCallback cb, void *privdata) {
    listIter li;
    listNode *ln;

    listRewind(server.clients, &li);
    while ((ln = listNext(&li))) {
        client *c = (client *)ln->value;

        // 回调可能删除当前或其他 client
        cb(c, privdata);

        // 【UAF】如果 cb 释放了 c，继续迭代会有问题
        // 即使 listNext 保存了 next，如果 cb 释放了多个 client...
    }
}

void killIdleCallback(client *c, void *privdata) {
    time_t timeout = *(time_t *)privdata;
    if ((server.unixtime - c->repl_ack_time) > timeout) {
        freeClient(c);
    }
}

void killIdleClients(time_t timeout) {
    forEachClient(killIdleCallback, &timeout);
}

// 【UAF Pattern 10】: 延迟释放列表处理不当
void processClientsToClose(void) {
    listIter li;
    listNode *ln;

    listRewind(server.clients_to_close, &li);
    while ((ln = listNext(&li))) {
        client *c = (client *)ln->value;

        // 先从 to_close 列表移除
        listDelNode(server.clients_to_close, ln);

        // 释放 client
        freeClient(c);

        // 【UAF】ln 已经被 listDelNode 释放，但 listNext 可能已经保存了错误的 next
    }
}

// 【UAF Pattern 11】: 复杂条件下的双重释放风险
void handleDisconnect(client *c) {
    static client *last_freed = NULL;

    // 防止双重释放的检查（但有 bug）
    if (c == last_freed) {
        serverLog(LL_WARNING, "Attempt to double free client %p", (void *)c);
        return;
    }

    if (c->flags & CLIENT_CLOSE_ASAP) {
        freeClient(c);
        last_freed = c;  // 【Bug】c 已释放，保存的是悬空指针
    }

    // 后续调用如果传入新分配的 client，可能地址相同
    // 导致错误跳过释放或双重释放
}

// 【UAF Pattern 12】: 原始 Redis 风格的复杂 replication cron
void replicationCron(void) {
    listIter li;
    listNode *ln;

    listRewind(server.slaves, &li);
    while ((ln = listNext(&li))) {
        client *slave = (client *)ln->value;

        // 检查 1: ONLINE 状态超时
        if (slave->replstate == SLAVE_STATE_ONLINE) {
            if (slave->flags & CLIENT_PRE_PSYNC)
                continue;
            if ((server.unixtime - slave->repl_ack_time) > server.repl_timeout) {
                serverLog(LL_WARNING, "Disconnecting timedout replica (streaming): %s",
                          getClientName(slave));
                freeClient(slave);
                // 【UAF】继续执行下面的检查
            }
        }

        // 检查 2: WAIT_BGSAVE_END 状态超时
        if (slave->replstate == SLAVE_STATE_WAIT_BGSAVE_END &&
            server.rdb_child_type == RDB_CHILD_TYPE_SOCKET) {
            if (slave->repl_last_partial_write != 0 &&
                (server.unixtime - slave->repl_last_partial_write) > server.repl_timeout) {
                serverLog(LL_WARNING, "Disconnecting timedout replica (full sync): %s",
                          getClientName(slave));
                freeClient(slave);
            }
        }

        // 检查 3: 发送 bulk 数据超时
        if (slave->replstate == SLAVE_STATE_SEND_BULK) {
            if ((server.unixtime - slave->repl_last_partial_write) > server.repl_timeout * 2) {
                serverLog(LL_WARNING, "Disconnecting timedout replica (bulk transfer): %s",
                          getClientName(slave));
                freeClient(slave);
            }
        }

        // 检查 4: 通用的 pending write 处理
        // 【UAF】如果上面任何一个 freeClient 被调用，这里都是 UAF
        if (slave->flags & CLIENT_PENDING_WRITE) {
            if (slave->conn && slave->conn->output->len > 0) {
                serverLog(LL_DEBUG, "Slave %s has %zu bytes pending",
                          slave->name, slave->conn->output->len);
            }
        }
    }
}

// 【UAF Pattern 13】: 命令处理中释放 current_client
void processCommand(client *c, command *cmd) {
    server.current_client = c;

    // 某些命令会触发 client 释放
    if (strcmp(cmd->name, "QUIT") == 0) {
        freeClient(c);
        // 忘记清除 current_client
    } else if (strcmp(cmd->name, "CLIENT") == 0 && cmd->argc > 1) {
        if (strcmp(cmd->argv[1], "KILL") == 0) {
            // 自杀
            if (cmd->argc > 2 && strcmp(cmd->argv[2], "SELF") == 0) {
                freeClient(c);
            }
        }
    }

    // 【UAF】如果是 QUIT 或 CLIENT KILL SELF，c 已被释放
    serverLog(LL_DEBUG, "Processed command %s for client %s", cmd->name, c->name);

    // 【UAF】current_client 也是悬空指针
    if (server.current_client && server.current_client->querybuf) {
        server.current_client->querybuf->len = 0;
    }

    server.current_client = NULL;
}

// 【UAF Pattern 14】: 信号处理器中的竞态条件模拟
volatile int got_signal = 0;

void simulateSignalHandler(void) {
    got_signal = 1;
    // 信号处理器可能在任何时候触发，设置 shutdown 标志
    server.shutdown_asap = 1;
}

void mainLoop(void) {
    listIter li;
    listNode *ln;

    while (!server.shutdown_asap) {
        listRewind(server.clients, &li);
        while ((ln = listNext(&li))) {
            client *c = (client *)ln->value;

            // 模拟信号在这里触发
            if (c->id == 2) {
                simulateSignalHandler();
            }

            // 信号处理器设置了 shutdown，开始清理
            if (server.shutdown_asap) {
                freeClient(c);
                // 【Bug】继续循环会访问其他已释放的 client
            }

            // 【UAF】
            serverLog(LL_DEBUG, "Processing client %s in main loop", c->name);
        }

        // 只运行一次用于演示
        break;
    }
}

// ============== 测试设置 ==============

client *createTestClient(int id, int replstate, int flags) {
    client *c = (client *)malloc(sizeof(*c));
    memset(c, 0, sizeof(*c));
    c->id = id;
    c->replstate = replstate;
    c->flags = flags;
    c->repl_ack_time = 800;
    c->repl_last_partial_write = 800;
    c->refcount = 1;
    c->authenticated = 1;
    snprintf(c->name, sizeof(c->name), "client-%d", id);
    c->conn = connCreate(id + 100);
    c->conn->owner = c;
    c->querybuf = bufferCreate(256);
    c->pubsub_channels = dictCreate();
    c->pending_commands = listCreate();
    return c;
}

void setupServer(void) {
    server.slaves = listCreate();
    server.clients = listCreate();
    server.clients_to_close = listCreate();
    server.clients_pending_write = listCreate();
    server.unixtime = 1000;
    server.repl_timeout = 60;
    server.rdb_child_type = RDB_CHILD_TYPE_SOCKET;
    server.shutdown_asap = 0;
    server.current_client = NULL;
    server.master = NULL;
    server.loading = 0;
    server.max_clients = 100;
}

void populateTestData(void) {
    // 创建一些 slaves
    for (int i = 0; i < 5; i++) {
        client *c = createTestClient(i, SLAVE_STATE_ONLINE, 0);
        listAddNodeTail(server.slaves, c);
        listAddNodeTail(server.clients, c);
    }

    // 创建一些普通 clients
    for (int i = 5; i < 10; i++) {
        int flags = (i % 2 == 0) ? CLIENT_PENDING_WRITE : 0;
        client *c = createTestClient(i, 0, flags);
        listAddNodeTail(server.clients, c);
    }

    // 设置一些 master-slave 关系
    client *master = createTestClient(100, 0, CLIENT_MASTER);
    server.master = master;
    listAddNodeTail(server.clients, master);

    // 让一些 slave 指向 master
    listIter li;
    listNode *ln;
    listRewind(server.slaves, &li);
    while ((ln = listNext(&li))) {
        client *slave = (client *)ln->value;
        if (slave->id < 3) {
            slave->master = master;
        }
    }
}

// ============== Main ==============

int main(void) {
    printf("=== Complex UAF Demonstration ===\n\n");

    setupServer();
    populateTestData();

    printf("Created %lu slaves and %lu total clients\n\n",
           server.slaves->len, server.clients->len);

    // 触发各种 UAF patterns
    printf("--- Running replicationCron (Pattern 12) ---\n");
    replicationCron();

    // 重新设置以测试其他模式
    setupServer();
    populateTestData();

    printf("\n--- Running processNestedClients (Pattern 4) ---\n");
    // 设置触发条件
    {
        listIter li;
        listNode *ln;
        listRewind(server.slaves, &li);
        while ((ln = listNext(&li))) {
            client *slave = (client *)ln->value;
            slave->flags |= CLIENT_CLOSE_ASAP;
            break;  // 只设置第一个
        }
    }
    processNestedClients();

    printf("\n=== Done ===\n");
    return 0;
}