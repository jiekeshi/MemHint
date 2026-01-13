export GOOGLE_APPLICATION_CREDENTIALS=/home/huihuihuang/Hint/huihui-472807-b8069b88637c.json
export PATH=/home/huihuihuang/Hint/codeql:$PATH
PROJECT_PATH=/home/huihuihuang/Hint/bugscpp/cpp_peglib/buggy-1  
OUTPUT_PATH=/home/huihuihuang/Hint/output/bugscpp/cpp_peglib/buggy-1/output
mkdir -p $OUTPUT_PATH
# python main.py --project $PROJECT_PATH --output $OUTPUT_PATH --merge
python main.py --project $PROJECT_PATH --output $OUTPUT_PATH