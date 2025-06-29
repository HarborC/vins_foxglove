#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "foxglove/FGVisualizer.h"

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
std::shared_ptr<FGVisualizer> viz;

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "/home/cat/projects/open_vins/config/ours/estimator_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  ros::init(argc, argv, "run_msckf_ros");
  auto nh = std::make_shared<ros::NodeHandle>("~");

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
  
  viz = std::make_shared<FGVisualizer>(sys);

  viz->setup_subscribers(nh, parser);

  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  // Final visualization
  viz->visualize_final();

  // Done!
  ros::shutdown();
  return EXIT_SUCCESS;
}
