# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build

# Include any dependencies generated for this target.
include CMakeFiles/Optim2GaussianSmooth.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Optim2GaussianSmooth.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Optim2GaussianSmooth.dir/flags.make

CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o: CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o.depend
CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o: CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o.cmake
CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o: ../Optim2GaussianSmooth.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o"
	cd /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir && /usr/local/bin/cmake -E make_directory /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir//.
	cd /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir//./Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o -D generated_cubin_file:STRING=/home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir//./Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o.cubin.txt -P /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir//Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o.cmake

# Object files for target Optim2GaussianSmooth
Optim2GaussianSmooth_OBJECTS =

# External object files for target Optim2GaussianSmooth
Optim2GaussianSmooth_EXTERNAL_OBJECTS = \
"/home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o"

Optim2GaussianSmooth: CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o
Optim2GaussianSmooth: CMakeFiles/Optim2GaussianSmooth.dir/build.make
Optim2GaussianSmooth: /usr/local/cuda-8.0/lib64/libcudart_static.a
Optim2GaussianSmooth: /usr/lib/x86_64-linux-gnu/librt.so
Optim2GaussianSmooth: CMakeFiles/Optim2GaussianSmooth.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Optim2GaussianSmooth"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Optim2GaussianSmooth.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Optim2GaussianSmooth.dir/build: Optim2GaussianSmooth

.PHONY : CMakeFiles/Optim2GaussianSmooth.dir/build

CMakeFiles/Optim2GaussianSmooth.dir/requires:

.PHONY : CMakeFiles/Optim2GaussianSmooth.dir/requires

CMakeFiles/Optim2GaussianSmooth.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Optim2GaussianSmooth.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Optim2GaussianSmooth.dir/clean

CMakeFiles/Optim2GaussianSmooth.dir/depend: CMakeFiles/Optim2GaussianSmooth.dir/Optim2GaussianSmooth_generated_Optim2GaussianSmooth.cu.o
	cd /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build /home/tly/myProgram/cudaProgram/GaussianSmooth/Optim2GaussianSmooth/build/CMakeFiles/Optim2GaussianSmooth.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Optim2GaussianSmooth.dir/depend

