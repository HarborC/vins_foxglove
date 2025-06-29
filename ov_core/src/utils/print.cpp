/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "print.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace ov_core;

// Need to define the static variable for everything to work
Printer::PrintLevel Printer::current_print_level = PrintLevel::INFO;

void Printer::setPrintLevel(const std::string &level) {
  if (level == "ALL")
    setPrintLevel(PrintLevel::ALL);
  else if (level == "OVDEBUG")
    setPrintLevel(PrintLevel::OVDEBUG);
  else if (level == "INFO")
    setPrintLevel(PrintLevel::INFO);
  else if (level == "WARNING")
    setPrintLevel(PrintLevel::WARNING);
  else if (level == "ERROR")
    setPrintLevel(PrintLevel::ERROR);
  else if (level == "SILENT")
    setPrintLevel(PrintLevel::SILENT);
  else {
    std::cout << "Invalid print level requested: " << level << std::endl;
    std::cout << "Valid levels are: ALL, OVDEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void Printer::setPrintLevel(PrintLevel level) {
  Printer::current_print_level = level;
  std::cout << "Setting printing level to: ";
  switch (current_print_level) {
  case PrintLevel::ALL:
    std::cout << "ALL";
    break;
  case PrintLevel::OVDEBUG:
    std::cout << "OVDEBUG";
    break;
  case PrintLevel::INFO:
    std::cout << "INFO";
    break;
  case PrintLevel::WARNING:
    std::cout << "WARNING";
    break;
  case PrintLevel::ERROR:
    std::cout << "ERROR";
    break;
  case PrintLevel::SILENT:
    std::cout << "SILENT";
    break;
  default:
    std::cout << std::endl;
    std::cout << "Invalid print level requested: " << level << std::endl;
    std::cout << "Valid levels are: ALL, OVDEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << std::endl;
}

void Printer::debugPrint(PrintLevel level, const char location[], const char line[], const char *format, ...) {
  // Only print for the current debug level
  if (static_cast<int>(level) < static_cast<int>(Printer::current_print_level)) {
    return;
  }

  // Print the location info first for our debug output
  // Truncate the filename to the max size for the filepath
  if (static_cast<int>(Printer::current_print_level) <= static_cast<int>(Printer::PrintLevel::OVDEBUG)) {
    std::string path(location);
    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
    if (base_filename.size() > MAX_FILE_PATH_LEGTH) {
      printf("%s", base_filename.substr(base_filename.size() - MAX_FILE_PATH_LEGTH, base_filename.size()).c_str());
    } else {
      printf("%s", base_filename.c_str());
    }
    printf(":%s ", line);
  }

  // Print the rest of the args
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

void initSpdlog(std::string node_name, std::string &log_path, bool alsologtostderr) {
  // 创建文件日志 sink
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      log_path + "/log_info.log", true);

  // 设置控制台输出 sink（带有颜色的输出）
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // 如果需要将日志同时输出到 stderr
  if (alsologtostderr) {
    console_sink->set_level(spdlog::level::info);
  } else {
    console_sink->set_level(spdlog::level::off); // 禁止控制台输出
  }

  // 设置文件输出的日志级别
  file_sink->set_level(spdlog::level::info); // 记录 INFO 及以上级别日志到文件

  // 创建一个 logger，将多个 sink 组合在一起
  spdlog::sinks_init_list sink_list = {file_sink, console_sink};
  auto logger = std::make_shared<spdlog::logger>(node_name, sink_list.begin(),
                                                 sink_list.end());

  // 设置日志格式
  logger->set_pattern("[%H:%M:%S.%e][%l][%g:%#] %v");

  // 注册 logger
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);

  // 设置日志级别
  spdlog::set_level(spdlog::level::info); // 默认级别为 INFO
  spdlog::flush_on(spdlog::level::info); // 每条日志输出后立即刷新到文件
}
