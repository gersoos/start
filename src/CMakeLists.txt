set(WITH_welcome        ON  CACHE BOOL "with welcome app")
set(WITH_welcome_qt     ON  CACHE BOOL "with welcome_qt app")

if(WITH_welcome)
	add_subdirectory(welcome)
endif()

if(WITH_welcome_qt)
	add_subdirectory(welcome_qt)
endif()
