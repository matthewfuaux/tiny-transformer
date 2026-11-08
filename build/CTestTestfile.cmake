# CMake generated Testfile for 
# Source directory: /home/matt/code/personal/tiny-transformer/tiny-transformer
# Build directory: /home/matt/code/personal/tiny-transformer/tiny-transformer/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_elementwise]=] "/home/matt/code/personal/tiny-transformer/tiny-transformer/build/test_elementwise")
set_tests_properties([=[test_elementwise]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;49;add_test;/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;52;add_kernel_test;/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;0;")
add_test([=[test_softmax]=] "/home/matt/code/personal/tiny-transformer/tiny-transformer/build/test_softmax")
set_tests_properties([=[test_softmax]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;49;add_test;/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;53;add_kernel_test;/home/matt/code/personal/tiny-transformer/tiny-transformer/CMakeLists.txt;0;")
