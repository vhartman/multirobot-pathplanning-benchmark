#include <chrono>
#include <iostream>
#include <pybind11/embed.h> // Pybind11 embed mode
#include <pybind11/numpy.h>

namespace py = pybind11;

int main() {
  // Initialize the Python interpreter
  py::scoped_interpreter guard{};

  try {
    // Import the Python module
    py::module_ my_module = py::module_::import(
        "multi_robot_multi_goal_planning.problems.rai_envs");

    // Get the Python class
    // py::object RaiTwoDimEnv = my_module.attr("rai_two_dim_env");
    py::object RaiTwoDimEnv = my_module.attr("rai_ur10_arm_box_stack_env");

    // Create an instance of the Python class
    py::object env = RaiTwoDimEnv();

    // env.attr("show")(true);

    const auto start_pose = env.attr("start_pos");
    const auto start_mode = env.attr("start_mode");

    const pybind11::array_t<double> start_pose_arr = start_pose.attr("state")();

    auto buffer = start_pose_arr.request();
    double *ptr = static_cast<double *>(buffer.ptr);

    // for (int i = 0; i < buffer.size; i++) {
    //   std::cout << ptr[i] << " ";
    // }

    // env.attr("is_collision_free")(start_pose, start_mode);

    std::vector<py::object> modes{start_mode};

    std::cout << "Attempting to generate new modes" << std::endl;
    for (uint i = 0; i < 500; ++i) {
      const auto mode = modes[rand() % modes.size()];
      const py::list next_modes_ids =
          env.attr("get_valid_next_task_combinations")(mode);

      if (next_modes_ids.size() > 0) {
        const int ind = rand() % next_modes_ids.size();

        const auto active_task =
            env.attr("get_active_task")(mode, next_modes_ids[ind]);

        const auto goal_config =
            env.attr("sample_goal_configuration")(mode, active_task);

        if (Py_IsTrue(env.attr("is_collision_free")(goal_config, mode))) {
          if (Py_IsTrue(env.attr("is_terminal_mode")(mode))) {
          } else {
            const auto next_mode = env.attr("get_next_mode")(goal_config, mode);
            modes.push_back(next_mode);
          }
        }
      }
    }

    std::cout << "Found " << modes.size() << " modes" << std::endl;

    std::cout << "Starting benhmark:" << std::endl;

    // Benchmark start
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call function 10,000 times
    const int N = 10000;
    for (int i = 0; i < N; i++) {
      const auto mode = modes[rand() % modes.size()];
      const auto pose = env.attr("sample_config_uniform_in_limits")();
      env.attr("is_collision_free")(pose, mode);

    //   env.attr("show")(true);
    }

    // Benchmark end
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print results
    std::cout << "Elapsed time for 10,000 calls: " << elapsed.count()
              << " seconds" << std::endl;
    std::cout << "Average time per call: " << (1. * elapsed.count() / N * 1000)
              << " ms" << std::endl;

    // env.attr("show")(true);
  } catch (const py::error_already_set &e) {
    std::cerr << "Python error: " << e.what() << std::endl;
  }

  return 0;
}
