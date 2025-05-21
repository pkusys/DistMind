add_subdirectory(generate_file)
add_subdirectory(deploy_file)

configure_file(generate_model_list.py ${EXECUTABLE_OUTPUT_PATH}/generate_model_list.py COPYONLY)
configure_file(generate_model_distribution.py ${EXECUTABLE_OUTPUT_PATH}/generate_model_distribution.py COPYONLY)