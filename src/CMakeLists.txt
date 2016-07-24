set(WITH_welcome        ON  CACHE BOOL "with welcome app")
set(WITH_welcome_qt     ON  CACHE BOOL "with welcome_qt app")
set(WITH_welcome_cv     ON  CACHE BOOL "with welcome_cv app")
set(WITH_lane           ON  CACHE BOOL "with lane app")


if(WITH_welcome)
	add_subdirectory(welcome)
endif()

if(WITH_welcome_qt)
	add_subdirectory(welcome_qt)
endif()

if(WITH_welcome_cv)
	add_subdirectory(welcome_cv)
endif()

if(WITH_lane)
	add_subdirectory(lane)
endif()