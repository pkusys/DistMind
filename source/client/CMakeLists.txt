configure_file(common.py ${LIBRARY_OUTPUT_PATH}/common.py COPYONLY)
configure_file(client_template.py ${LIBRARY_OUTPUT_PATH}/client_template.py COPYONLY)
configure_file(client_fixed_rate.py ${LIBRARY_OUTPUT_PATH}/client_fixed_rate.py COPYONLY)
configure_file(client_max_rate.py ${LIBRARY_OUTPUT_PATH}/client_max_rate.py COPYONLY)
configure_file(client.py ${LIBRARY_OUTPUT_PATH}/client.py COPYONLY)
configure_file(client_fixed_inference.py ${EXECUTABLE_OUTPUT_PATH}/client_fixed_inference.py COPYONLY)
configure_file(client_max_inference.py ${EXECUTABLE_OUTPUT_PATH}/client_max_inference.py COPYONLY)
configure_file(client_max_train.py ${EXECUTABLE_OUTPUT_PATH}/client_max_train.py COPYONLY)
configure_file(client_one.py ${EXECUTABLE_OUTPUT_PATH}/client_one.py COPYONLY)
configure_file(client_concurrent.py ${EXECUTABLE_OUTPUT_PATH}/client_concurrent.py COPYONLY)