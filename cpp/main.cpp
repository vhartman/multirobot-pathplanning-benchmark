#include <iostream>
#include <pybind11/embed.h> // Pybind11 embed mode
#include <pybind11/numpy.h>
#include <chrono>

namespace py = pybind11;

int main() {
  // Initialize the Python interpreter
  py::scoped_interpreter guard{};

  try {
    // Import the Python module
    py::module_ my_module = py::module_::import(
        "multi_robot_multi_goal_planning.problems.rai_envs");

    // Get the Python class
    py::object RaiTwoDimEnv = my_module.attr("rai_two_dim_env");
    // py::object RaiTwoDimEnv = my_module.attr("rai_ur10_arm_box_stack_env");

    // Create an instance of the Python class
    py::object py_instance = RaiTwoDimEnv();

    // py_instance.attr("show")(true);

    const auto start_pose = py_instance.attr("start_pos");
    const auto start_mode = py_instance.attr("start_mode");

    const pybind11::array_t<double> start_pose_arr = start_pose.attr("state")();
    
    auto buffer = start_pose_arr.request();
    double* ptr = static_cast<double*>(buffer.ptr);

    for (int i = 0; i < buffer.size; i++) {
        std::cout << ptr[i] << " ";
    }

    py_instance.attr("is_collision_free")(start_pose, start_mode);

    // Benchmark start
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call function 10,000 times
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        const auto pose = py_instance.attr("sample_config_uniform_in_limits")();
        py_instance.attr("is_collision_free")(pose, start_mode);
    }

    // Benchmark end
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print results
    std::cout << "Elapsed time for 10,000 calls: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average time per call: " << (1. * elapsed.count() / N * 1000) << " ms" << std::endl;


    // py_instance.attr("show")(true);
  } catch (const py::error_already_set &e) {
    std::cerr << "Python error: " << e.what() << std::endl;
  }

  return 0;
}
