# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/qlw/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/qlw/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/qlw/cx/cs344/Problem Sets/Problem Set 1"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build"

# Include any dependencies generated for this target.
include CMakeFiles/HW1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/HW1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/HW1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HW1.dir/flags.make

CMakeFiles/HW1.dir/main.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/main.o: /home/qlw/cx/cs344/Problem\ Sets/Problem\ Set\ 1/main.cpp
CMakeFiles/HW1.dir/main.o: CMakeFiles/HW1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/HW1.dir/main.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/HW1.dir/main.o -MF CMakeFiles/HW1.dir/main.o.d -o CMakeFiles/HW1.dir/main.o -c "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/main.cpp"

CMakeFiles/HW1.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/HW1.dir/main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/main.cpp" > CMakeFiles/HW1.dir/main.i

CMakeFiles/HW1.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/main.cpp" -o CMakeFiles/HW1.dir/main.s

CMakeFiles/HW1.dir/reference_calc.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/reference_calc.o: /home/qlw/cx/cs344/Problem\ Sets/Problem\ Set\ 1/reference_calc.cpp
CMakeFiles/HW1.dir/reference_calc.o: CMakeFiles/HW1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/HW1.dir/reference_calc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/HW1.dir/reference_calc.o -MF CMakeFiles/HW1.dir/reference_calc.o.d -o CMakeFiles/HW1.dir/reference_calc.o -c "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/reference_calc.cpp"

CMakeFiles/HW1.dir/reference_calc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/HW1.dir/reference_calc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/reference_calc.cpp" > CMakeFiles/HW1.dir/reference_calc.i

CMakeFiles/HW1.dir/reference_calc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/reference_calc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/reference_calc.cpp" -o CMakeFiles/HW1.dir/reference_calc.s

CMakeFiles/HW1.dir/compare.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/compare.o: /home/qlw/cx/cs344/Problem\ Sets/Problem\ Set\ 1/compare.cpp
CMakeFiles/HW1.dir/compare.o: CMakeFiles/HW1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/HW1.dir/compare.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/HW1.dir/compare.o -MF CMakeFiles/HW1.dir/compare.o.d -o CMakeFiles/HW1.dir/compare.o -c "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/compare.cpp"

CMakeFiles/HW1.dir/compare.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/HW1.dir/compare.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/compare.cpp" > CMakeFiles/HW1.dir/compare.i

CMakeFiles/HW1.dir/compare.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/compare.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/compare.cpp" -o CMakeFiles/HW1.dir/compare.s

# Object files for target HW1
HW1_OBJECTS = \
"CMakeFiles/HW1.dir/main.o" \
"CMakeFiles/HW1.dir/reference_calc.o" \
"CMakeFiles/HW1.dir/compare.o"

# External object files for target HW1
HW1_EXTERNAL_OBJECTS =

HW1: CMakeFiles/HW1.dir/main.o
HW1: CMakeFiles/HW1.dir/reference_calc.o
HW1: CMakeFiles/HW1.dir/compare.o
HW1: CMakeFiles/HW1.dir/build.make
HW1: CMakeFiles/HW1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable HW1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HW1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HW1.dir/build: HW1
.PHONY : CMakeFiles/HW1.dir/build

CMakeFiles/HW1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HW1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HW1.dir/clean

CMakeFiles/HW1.dir/depend:
	cd "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/qlw/cx/cs344/Problem Sets/Problem Set 1" "/home/qlw/cx/cs344/Problem Sets/Problem Set 1" "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build" "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build" "/home/qlw/cx/cs344/Problem Sets/Problem Set 1/build/CMakeFiles/HW1.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/HW1.dir/depend

