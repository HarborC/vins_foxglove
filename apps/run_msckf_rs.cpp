// 引入智能指针支持
#include <memory>

// 引入自定义的 VIO 系统核心模块和可视化模块
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "foxglove/FGVisualizer.h"

// 使用 ov_msckf 命名空间
using namespace ov_msckf;

// 定义全局 VIO 系统指针和可视化工具指针
std::shared_ptr<VioManager> sys;
std::shared_ptr<FGVisualizer> viz;

// 主函数入口
int main(int argc, char **argv) {

  bool is_debug = false; // 是否开启调试模式标志

  // 默认配置文件路径（使用项目目录宏 PROJ_DIR）
  std::string config_path = std::string(PROJ_DIR) + "/config/rs_t265/estimator_config.yaml";

  // 如果命令行参数个数大于 1，则开启调试模式
  if (argc > 1) {
    is_debug = true;
  }

  // 加载 YAML 配置文件
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  // 初始化 VIO 管理器参数
  VioManagerOptions params;
  params.print_and_load(parser);             // 打印并加载配置参数
  params.use_multi_threading_subs = true;    // 启用多线程订阅器（提高性能）

  // 创建 VIO 系统实例
  sys = std::make_shared<VioManager>(params);

  // 创建可视化工具实例，并传入 VIO 系统
  viz = std::make_shared<FGVisualizer>(sys);
  viz->is_debug = is_debug;  // 设置是否为调试模式

  // 启动可视化运行主循环
  viz->runRealsenseIO();

  // 最后阶段的可视化（如轨迹、地图等）
  viz->visualize_final();

  // 程序正常退出
  return EXIT_SUCCESS;
}
