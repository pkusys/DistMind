add_subdirectory(cpp_extension)

configure_file(deploy_file.py ${EXECUTABLE_OUTPUT_PATH}/deploy_file.py COPYONLY)