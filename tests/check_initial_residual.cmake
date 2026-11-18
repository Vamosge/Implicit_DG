if(NOT DEFINED SOLVER OR NOT DEFINED EXPECTED_PREFIX)
  message(FATAL_ERROR "SOLVER and EXPECTED_PREFIX are required")
endif()

execute_process(
  COMMAND "${SOLVER}" --residual-check
  RESULT_VARIABLE solver_result
  OUTPUT_VARIABLE solver_output
  ERROR_VARIABLE solver_error)

if(NOT solver_result EQUAL 0)
  message(FATAL_ERROR
    "Residual check failed with exit code ${solver_result}\n"
    "stdout:\n${solver_output}\n"
    "stderr:\n${solver_error}")
endif()

set(expected_text
  "[RESIDUAL CHECK] status=ok max_l2_scaled=${EXPECTED_PREFIX}")
string(FIND "${solver_output}" "${expected_text}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "Initial residual changed; expected prefix '${EXPECTED_PREFIX}'.\n"
    "Output:\n${solver_output}")
endif()

message(STATUS
  "Initial residual matches trusted prefix ${EXPECTED_PREFIX}")
