set(WITH_welcome        ON  CACHE BOOL "with welcome app")
set(WITH_welcome_qt     ON  CACHE BOOL "with welcome_qt app")
set(WITH_welcome_cv     ON  CACHE BOOL "with welcome_cv app")


if(WITH_welcome)
	add_subdirectory(welcome)
endif()

if(WITH_welcome_qt)
	add_subdirectory(welcome_qt)
endif()

if(WITH_welcome_cv)
	add_subdirectory(welcome_cv)
endif()
