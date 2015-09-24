set(TARGET torus_benchmark)
set(MY_LIB_SRC "../../src")
set(MY_LIB_INC "../../include")

include_directories(${MY_LIB_INC})

# Listing all files to be included for the target.
# New files must be added to this manually.
set(TARGET_SRC
    ${MY_LIB_SRC}/all_data.cc
    ${MY_LIB_SRC}/curlfunction.cc
    ${MY_LIB_SRC}/backgroundfield.cc
    ${MY_LIB_SRC}/inputtools.cc
    ${MY_LIB_SRC}/mydofrenumbering.cc
    ${MY_LIB_SRC}/mypreconditioner.cc
    ${MY_LIB_SRC}/mypolynomials.cc
    ${MY_LIB_SRC}/myfe_nedelec.cc
    ${MY_LIB_SRC}/myvectortools.cc
    ${MY_LIB_SRC}/outputtools.cc
    ${MY_LIB_SRC}/forwardsolver.cc
    src/${TARGET}
    )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
