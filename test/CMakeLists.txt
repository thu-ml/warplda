set (LINK_LIBS ${LINK_LIBS} pthread)

macro (test_program target)
	add_executable (${target} "${target}.cpp")
	target_link_libraries (${target} ${LINK_LIBS})
endmacro (test_program)

macro (do_test target arg result)
	add_test (${target}-${arg} ${target} ${arg})
	set_tests_properties (${target}-${arg} PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro (do_test)

test_program (basic)
do_test (basic "" "OK")
