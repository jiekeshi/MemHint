export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
# PROJECT_PATH=/home/huihuihuang/Hint/bugscpp/cpp_peglib/buggy-1  
# OUTPUT_PATH=/home/huihuihuang/Hint/output/bugscpp/cpp_peglib/buggy-1/output
PROJECT_PATH=/home/huihuihuang/Hint/defect4c_clone_proj/njs
OUTPUT_PATH=/home/huihuihuang/Hint/output/defect4c_clone_proj/njs/output
mkdir -p $OUTPUT_PATH
# python main.py --project $PROJECT_PATH --output $OUTPUT_PATH --merge
python main.py --project $PROJECT_PATH --output $OUTPUT_PATH --single-source "/home/huihuihuang/Hint/defect4c_clone_proj/njs/src/njs_function.c" --cpp-queries-dir "/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries"
# python main.py --project $PROJECT_PATH --output $OUTPUT_PATH --single-source "/home/huihuihuang/Hint/defect4c_clone_proj/njs/src/njs_function.c" --cpp-queries-dir "/home/huihuihuang/Hint/codeql/qlpacks/codeql/cpp-queries" --no-reuse-db