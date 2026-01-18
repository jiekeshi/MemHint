export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
CPP_QUERIES_DIR="/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries"

##########
# Redis #
##########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/redis
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/redis/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/redis/src/replication.c" "/home/huihuihuang/Hint/cloned_proj/redis/src/redis-benchmark.c"

##########
# Htop #
##########
# PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/htop
# OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/htop/output_single_source
# mkdir -p $OUTPUT_PATH
# python main.py --debug --project $PROJECT_PATH --output $OUTPUT_PATH \
#     --cpp-queries-dir $CPP_QUERIES_DIR \
#     --single-source "/home/huihuihuang/Hint/cloned_proj/htop/Settings.c" "/home/huihuihuang/Hint/cloned_proj/htop/XUtils.c"

##########
# tmux_pr3937 #
##########
PROJECT_PATH=/home/huihuihuang/Hint/cloned_proj/tmux_pr3937
OUTPUT_PATH=/home/huihuihuang/Hint/output/cloned_proj/tmux_pr3937/output_single_source
mkdir -p $OUTPUT_PATH
python main.py --debug --project $PROJECT_PATH --output $OUTPUT_PATH \
    --cpp-queries-dir $CPP_QUERIES_DIR \
    --single-source "/home/huihuihuang/Hint/cloned_proj/tmux_pr3937/arguments.c"