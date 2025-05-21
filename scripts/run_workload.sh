NUM_REQUEST=64
GROUP_SIZE=1
ENABLE_LOOP=loop
ZIPF_PARAM=0.0
THROUGHPUT=1
ENABLE_UNIFORM=uniform

python build/bin/workload.py build/resource/model_list.txt build/resource/request_list.txt $NUM_REQUEST $GROUP_SIZE $ENABLE_LOOP $ZIPF_PARAM $THROUGHPUT $ENABLE_UNIFORM