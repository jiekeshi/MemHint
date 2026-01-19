export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
CPP_QUERIES_DIR="/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries"

##########
# Redis #
##########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/redis
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/redis/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/redis/src/replication.c" "/home/huihuihuang/Hint/cloned_proj/redis/src/redis-benchmark.c"

##########
# Htop #
##########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/htop
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/htop/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/htop/Settings.c" "/home/huihuihuang/Hint/cloned_proj/htop/XUtils.c"

##########
# tmux issue3937 using tmux_pr3941_before#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3937_pr3941_before/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/arguments.c"

##########
# tmux issue3938 using tmux_pr3941_before#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3938_pr3941_before/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-command-prompt.c"


##########
# tmux issue3939 using tmux_pr3941_before#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3939_pr3941_before/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/cmd-confirm-before.c"

##########
# tmux issue3940 using tmux_pr3941_before#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3940_pr3941_before/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout-custom.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3941_before/layout.c"

##########
# tmux issue3949 using tmux_pr3972_before#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3949_pr3972_before/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_pr3972_before/arguments.c"

##########
# tmux issue3950 using tmux_20240420#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3950_20240420/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/cmd-confirm-before.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"

##########
# tmux issue3951 using tmux_20240420#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3951_20240420/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/compat/reallocarray.c"

##########
# tmux issue3952 using tmux_20240420#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20240420
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue3952_20240420/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/status.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20240420/xmalloc.c"

##########
# tmux issue4298 using tmux_20241213#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20241213
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4298_20241213/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20241213/mode-tree.c"

##########
# tmux issue4394 using tmux_20250303#
#########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_20250303
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_issue4394_20250303/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/window.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/utf8.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/xmalloc.c" "/home/huihuihuang/Hint/cloned_proj/tmux_20250303/compat/reallocarray.c"

##########
# zstd issue4112 using zstd_pr4115_before#
#########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before
OUTPUT_PATH=/home/huihuihuang/Hint/cloned_proj/zstd_issue4112_pr4115_before/output_single_source
mkdir -p $OUTPUT_PATH
python main.py --debug --no-reuse-db --project $PROJECT_PATH --output $OUTPUT_PATH \
    --cpp-queries-dir $CPP_QUERIES_DIR \
    --single-source "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/compress/zstd_compress.c" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/allocations.h" "/home/huihuihuang/Hint/cloned_proj/zstd_pr4115_before/lib/common/zstd_deps.h"
