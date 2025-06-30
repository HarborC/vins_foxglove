#include <iostream>
#include <sstream>

#if defined(_WIN32)
    #include <windows.h>
    #include <psapi.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <fstream>
    #include <string>
    #include <unistd.h>
#endif

// 获取当前进程内存占用（MB）
inline double get_process_memory_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.WorkingSetSize / 1024.0 / 1024.0;

#elif defined(__APPLE__)
    task_basic_info info;
    mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size / 1024.0 / 1024.0;
    }
    return 0.0;

#elif defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            long kb;
            iss >> label >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
#endif
}

// 获取系统总内存（MB）
inline double get_total_memory_mb() {
#if defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return statex.ullTotalPhys / 1024.0 / 1024.0;

#elif defined(__APPLE__)
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
    return memsize / 1024.0 / 1024.0;

#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.substr(0, 8) == "MemTotal") {
            std::istringstream iss(line);
            std::string label;
            long kb;
            iss >> label >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
#endif
}

// 获取系统剩余可用内存（MB）
inline double get_free_memory_mb() {
#if defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return statex.ullAvailPhys / 1024.0 / 1024.0;

#elif defined(__APPLE__)
    mach_port_t host_port = mach_host_self();
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics_data_t vm_stat;

    if (host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &count) == KERN_SUCCESS) {
        int64_t free_pages = static_cast<int64_t>(vm_stat.free_count + vm_stat.inactive_count);
        int64_t page_size;
        size_t size = sizeof(page_size);
        sysctlbyname("hw.pagesize", &page_size, &size, nullptr, 0);
        return (free_pages * page_size) / 1024.0 / 1024.0;
    }
    return 0.0;

#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.substr(0, 13) == "MemAvailable:") {
            std::istringstream iss(line);
            std::string label;
            long kb;
            iss >> label >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
#endif
}

// 打印完整内存信息
inline void print_memory_usage() {
    double used_mb = get_process_memory_mb();
    double total_mb = get_total_memory_mb();
    double free_mb  = get_free_memory_mb();

    std::cout << "当前进程内存: " << used_mb << " MB"
              << " / 系统总内存: " << total_mb / 1024.0 << " GB"
              << " / 剩余可用: " << free_mb / 1024.0 << " GB"
              << std::endl;
}
