#!/usr/bin/env ruby
# Xcode project configuration script for Cactus iOS tests
# Sets up test files, libraries, and build settings

require 'xcodeproj'

def fail_with(message)
  STDERR.puts "Error: #{message}"
  exit 1
end

# Load environment variables
project_root = ENV['PROJECT_ROOT']
tests_root = ENV['TESTS_ROOT']
cactus_root = ENV['CACTUS_ROOT']
apple_root = ENV['APPLE_ROOT']
project_path = ENV['XCODEPROJ_PATH']

fail_with("PROJECT_ROOT not set") unless project_root
fail_with("TESTS_ROOT not set") unless tests_root
fail_with("CACTUS_ROOT not set") unless cactus_root
fail_with("XCODEPROJ_PATH not set") unless project_path
fail_with("Xcode project not found") unless File.exist?(project_path)

begin
  project = Xcodeproj::Project.open(project_path)
rescue => e
  fail_with("Failed to open Xcode project: #{e.message}")
end

target = project.targets.first
fail_with("No targets found in Xcode project") unless target

# Setup Tests group
tests_group = project.main_group.find_subpath('Tests', true)
tests_group.set_path(tests_root)
tests_group.set_source_tree('<absolute>')

# Test files with their renamed main functions (to avoid conflicts)
test_files = {
  'test_kernel.cpp' => 'test_kernel_main',
  'test_graph.cpp' => 'test_graph_main',
  'test_kv_cache.cpp' => 'test_kv_cache_main',
  'test_engine.cpp' => 'test_engine_main',
  'test_performance.cpp' => 'test_performance_main',
  'test_utils.cpp' => nil
}

# Add test files to project
test_files.each do |filename, renamed_main|
  file_path = File.join(tests_root, filename)
  existing_file = tests_group.files.find { |f| f.path == filename || f.real_path&.to_s == file_path }

  file_ref = if existing_file
    existing_file
  else
    ref = tests_group.new_reference(file_path)
    ref.set_source_tree('<absolute>')
    ref
  end

  build_file = target.source_build_phase.files.find { |bf| bf.file_ref == file_ref }
  build_file = target.source_build_phase.add_file_reference(file_ref) unless build_file

  build_file.settings = { 'COMPILER_FLAGS' => "-Dmain=#{renamed_main}" } if renamed_main
end

# Add test_utils.h header
test_utils_h = File.join(tests_root, 'test_utils.h')
unless tests_group.files.any? { |f| f.path == 'test_utils.h' }
  file_ref = tests_group.new_reference(test_utils_h)
  file_ref.set_source_tree('<absolute>')
end

# Link static libraries
frameworks_group = project.main_group.find_subpath('Frameworks', true)
frameworks_group.set_source_tree('<group>')

lib_device = File.join(apple_root, 'libcactus-device.a')
lib_simulator = File.join(apple_root, 'libcactus-simulator.a')

[lib_device, lib_simulator].each do |lib_path|
  lib_name = File.basename(lib_path)
  existing_lib = frameworks_group.files.find { |f| f.path == lib_name }

  unless existing_lib
    file_ref = frameworks_group.new_reference(lib_path)
    file_ref.set_source_tree('<absolute>')
    target.frameworks_build_phase.add_file_reference(file_ref)
  end
end

# Configure build settings
target.build_configurations.each do |config|
  # Header search paths
  config.build_settings['HEADER_SEARCH_PATHS'] ||= ['$(inherited)']
  config.build_settings['HEADER_SEARCH_PATHS'] << tests_root unless config.build_settings['HEADER_SEARCH_PATHS'].include?(tests_root)
  config.build_settings['HEADER_SEARCH_PATHS'] << cactus_root unless config.build_settings['HEADER_SEARCH_PATHS'].include?(cactus_root)

  ['graph', 'engine', 'kernel', 'ffi', 'models'].each do |subdir|
    subdir_path = File.join(cactus_root, subdir)
    config.build_settings['HEADER_SEARCH_PATHS'] << subdir_path unless config.build_settings['HEADER_SEARCH_PATHS'].include?(subdir_path)
  end

  # Library search paths
  config.build_settings['LIBRARY_SEARCH_PATHS'] ||= ['$(inherited)']
  config.build_settings['LIBRARY_SEARCH_PATHS'] << apple_root unless config.build_settings['LIBRARY_SEARCH_PATHS'].include?(apple_root)

  # Compiler settings
  config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++20'
  config.build_settings['CLANG_CXX_LIBRARY'] = 'libc++'
  config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] ||= ['$(inherited)']
  config.build_settings['OTHER_CPLUSPLUSFLAGS'] << '-O3'
end

begin
  project.save
rescue => e
  fail_with("Failed to save Xcode project: #{e.message}")
end
