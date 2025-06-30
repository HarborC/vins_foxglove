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

  bool is_debug = false;
  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "/home/cat/projects/open_vins/config/ours/estimator_config.yaml";
  if (argc > 1) {
    is_debug = true;
  }

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
  
  viz = std::make_shared<FGVisualizer>(sys);
  viz->is_debug = is_debug;
  viz->run();

  // Final visualization
  viz->visualize_final();

  // Done!
  return EXIT_SUCCESS;
}
