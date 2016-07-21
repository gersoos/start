set(WITH_welcome        ON  CACHE BOOL "with welcome app")

if(WITH_welcome)
	add_subdirectory(welcome)
endif()
