#pragma once

#include <memory>
#include <string>
#include <vector>

#include "utils/colors.h"

#define DELETE_LAST_LINE "\033[A\33[2K\r"
#define DELETE_CURRENT_LINE "\33[2K\r"
#define SCREEN_FLUSH std::fflush(stdout);

namespace ov_msckf {

class DashBoard {
public:
  using Ptr = std::shared_ptr<DashBoard>;

public:
  DashBoard();
  ~DashBoard();

  std::string getCurrentTime();

  void setDescription(const std::string &desc = "");

  template <typename T> void setNameAndValue(int id, const std::string &show_name, const T &show_value) {
    std::string show_value_str = std::to_string(show_value);
    setNameAndStringValue(id, show_name, show_value_str);
  }

  void setNameAndStringValue(int id, const std::string &show_name, const std::string &show_value);

  void print();

protected:
  std::string description = "";
  std::vector<std::string> show_names;
  std::vector<std::string> show_values;
  int last_stamped_mem_mb = 0;
  std::vector<std::string> show_values_colors = {WHITE, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN};
  std::vector<std::string> show_names_colors = {BOLDWHITE, BOLDRED, BOLDGREEN, BOLDYELLOW, BOLDBLUE, BOLDMAGENTA, BOLDCYAN};
};

} // namespace ov_msckf
