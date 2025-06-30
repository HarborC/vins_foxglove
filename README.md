# VINS (Visual-Inertial Navigation System)
该项目是一个基于 IMU 与双目相机的紧耦合状态估计系统，使用 MSCKF 框架进行状态估计，并使用 Foxglove 进行实时可视化。

# 📦 依赖安装
```
sudo apt update
sudo apt-get install -y libasio-dev libwebsocketpp-dev nlohmann-json3-dev
sudo apt-get install -y libeigen3-dev libopencv-dev libboost-all-dev autoconf automake libtool m4 libprotobuf-dev protobuf-compiler
sudo apt-get install -y cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev libceres-dev
```

# 🧱 编译项目
进入项目根目录，执行构建脚本：
```
cd vins_foxglove
sh build.sh
```
构建完成后，生成的可执行程序将在 build/apps/ 下。

# 🚀 启动算法
```
# 运行算法
./apps/run_msckf
```
程序将开始接收 IMU 与图像数据并进行状态估计。

# 📊 可视化（Foxglove）
1. 安装 Foxglove Studio
请前往官网下载安装：https://foxglove.dev/download
2. 配置可视化参数
请参考项目中的使用说明文档（misc/how_to_visualize.md）：
- 程序默认使用 8088 端口进行数据发布。
- 确保 Foxglove 端口与之匹配。
3. 加载可视化布局
我们提供了预配置的面板布局文件(misc/msckf2.json)：
- 打开 Foxglove 后导入该文件，即可看到实时状态、轨迹、图像与 IMU 信息。