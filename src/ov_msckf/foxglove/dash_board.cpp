#include "dash_board.h"
#include "utils/memory_utils.h"
#include "utils/print.h"

#include <iomanip>
#include <iostream>

namespace ov_msckf {

DashBoard::DashBoard() {
  show_names.resize(50);
  show_values.resize(50);
}

DashBoard::~DashBoard() { PRINT_INFO("DashBoard release"); }

void DashBoard::setNameAndStringValue(int id, const std::string &show_name, const std::string &show_value) {
  std::string show_value_str = show_value;
  int max_size = std::max(show_name.size(), show_value_str.size()) + 2;

  int add_show_name_size = max_size - show_name.size();
  int add_show_value_size = max_size - show_value_str.size();

  std::string new_show_name;
  if (add_show_name_size > 0) {
    int empty_num = (add_show_name_size / 2.0 + 0.5);
    new_show_name = std::string(empty_num, ' ') + show_name + std::string(empty_num, ' ');
  } else {
    new_show_name = show_name;
  }

  std::string new_show_value;
  if (add_show_value_size > 0) {
    int empty_num = (add_show_value_size / 2.0 + 0.5);
    new_show_value = std::string(empty_num, ' ') + show_value_str + std::string(empty_num, ' ');
  } else {
    new_show_value = show_value_str;
  }

  if (new_show_value.size() >= new_show_name.size()) {
    new_show_name += std::string(new_show_value.size() - new_show_name.size(), ' ');
  } else {
    new_show_value += std::string(new_show_name.size() - new_show_value.size(), ' ');
  }

  show_names[id + 2] = (new_show_name);
  show_values[id + 2] = (new_show_value);
}

void DashBoard::setDescription(const std::string &desc) { description = desc; }

std::string DashBoard::getCurrentTime() {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

void DashBoard::print() {

  std::string out_str_line_1, out_str_line_2;
  out_str_line_1.append("|");
  out_str_line_2.append("|");

  setNameAndStringValue(-2, "System time", getCurrentTime());

  double mem_used_mb = get_free_memory_mb();
  setNameAndValue(-1, "Memory left (GB)", mem_used_mb / 1024.0);

  int l_size = 1;
  for (int i = 0; i < show_names.size(); i++) {
    if (show_names[i] == "" || show_values[i] == "") {
      continue;
    }

    if (i < show_names_colors.size()) {
      out_str_line_1.append(show_names_colors[i]);
    } else {
      out_str_line_1.append(BOLDWHITE);
    }
    out_str_line_1.append(show_names[i]);
    out_str_line_1.append(RESET);
    out_str_line_1.append("|");
    l_size += show_names[i].size() + 1;

    if (i < show_values_colors.size()) {
      out_str_line_2.append(show_values_colors[i]);
    } else {
      out_str_line_2.append(WHITE);
    }
    out_str_line_2.append(show_values[i]);
    out_str_line_2.append(RESET);
    out_str_line_2.append("|");
  }

  if ((mem_used_mb - last_stamped_mem_mb < 1024) && last_stamped_mem_mb != 0) {
    std::cout << DELETE_CURRENT_LINE << DELETE_LAST_LINE << DELETE_LAST_LINE << DELETE_LAST_LINE << DELETE_LAST_LINE;
  } else {
    last_stamped_mem_mb = mem_used_mb;
  }

  std::string program_name = " MSCKF Dashboard ";
  int add_program_name_size = l_size - program_name.size();
  std::string new_program_name =
      std::string(add_program_name_size / 2, '=') + program_name + std::string(add_program_name_size - add_program_name_size / 2, '=');
  // std::cout << "\r\n" << std::endl;
  std::cout << BOLDWHITE << new_program_name << RESET << std::endl;
  std::cout << out_str_line_1 << RESET << std::endl;
  std::cout << out_str_line_2 << RESET << std::endl;
  std::cout << BOLDWHITE << description << RESET << std::endl;
  std::cout.flush();

  show_names.resize(50, "");
  show_values.resize(50, "");
  description = "";
}

} // namespace ov_msckf
