/**
 * @file shared_weight_table.h
 * @brief 基于 mmap + flock 的进程间共享权重表
 *
 * 多个 ChampSim 进程可通过同一个共享内存文件实时共享 Q-learning 权重。
 * 文件布局与 WeightsFileHeader + double[] 完全一致，因此可直接由
 * save_weights / load_weights 创建的 .bin 文件初始化。
 *
 * 锁策略：
 *   - flock(LOCK_SH)  用于读快照（多进程可并发读）
 *   - flock(LOCK_EX)  用于写增量（互斥写入）
 *   - 进程异常退出时内核自动释放锁
 */

#ifndef SHARED_WEIGHT_TABLE_H
#define SHARED_WEIGHT_TABLE_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <iostream>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>

class SharedWeightTable {
public:
    // 与 WeightsFileHeader 二进制兼容
#pragma pack(push, 1)
    struct Header {
        uint32_t magic;
        uint32_t version;
        int32_t  num_tilings;
        int32_t  grid_size_a;
        int32_t  grid_size_m;
        uint32_t memory_size;
        int32_t  num_actions;
        int32_t  num_way;
    };
#pragma pack(pop)

    static constexpr uint32_t MAGIC   = 0x524C5057;  // "RLPW"
    static constexpr uint32_t VERSION = 1;

    SharedWeightTable() = default;
    ~SharedWeightTable() { close(); }

    SharedWeightTable(const SharedWeightTable&) = delete;
    SharedWeightTable& operator=(const SharedWeightTable&) = delete;

    SharedWeightTable(SharedWeightTable&& other) noexcept
        : fd_(other.fd_), mapped_(other.mapped_),
          mapped_size_(other.mapped_size_), header_(other.header_),
          weights_ptr_(other.weights_ptr_), total_weights_(other.total_weights_)
    {
        other.fd_ = -1;
        other.mapped_ = nullptr;
        other.mapped_size_ = 0;
        other.header_ = nullptr;
        other.weights_ptr_ = nullptr;
        other.total_weights_ = 0;
    }

    SharedWeightTable& operator=(SharedWeightTable&& other) noexcept
    {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            mapped_ = other.mapped_;
            mapped_size_ = other.mapped_size_;
            header_ = other.header_;
            weights_ptr_ = other.weights_ptr_;
            total_weights_ = other.total_weights_;
            other.fd_ = -1;
            other.mapped_ = nullptr;
            other.mapped_size_ = 0;
            other.header_ = nullptr;
            other.weights_ptr_ = nullptr;
            other.total_weights_ = 0;
        }
        return *this;
    }

    /**
     * @brief 打开（映射）共享内存文件
     * @return 成功返回 true
     */
    bool open(const std::string& path, size_t total_weights,
              int32_t num_tilings, int32_t grid_size_a, int32_t grid_size_m,
              uint32_t memory_size, int32_t num_actions, int32_t num_way)
    {
        size_t expected_size = sizeof(Header) + total_weights * sizeof(double);

        fd_ = ::open(path.c_str(), O_RDWR);
        if (fd_ < 0) {
            std::cerr << "[SharedWeightTable] 无法打开共享内存文件: "
                      << path << std::endl;
            return false;
        }

        struct stat st{};
        if (fstat(fd_, &st) < 0 ||
            static_cast<size_t>(st.st_size) < expected_size) {
            std::cerr << "[SharedWeightTable] 文件大小不匹配: "
                      << st.st_size << " (期望 >= " << expected_size
                      << ")" << std::endl;
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        mapped_ = mmap(nullptr, expected_size,
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            std::cerr << "[SharedWeightTable] mmap 失败" << std::endl;
            ::close(fd_);
            fd_ = -1;
            mapped_ = nullptr;
            return false;
        }

        mapped_size_ = expected_size;
        header_ = reinterpret_cast<Header*>(mapped_);

        if (header_->magic != MAGIC || header_->version != VERSION) {
            std::cerr << "[SharedWeightTable] 文件头 magic/version 不匹配"
                      << std::endl;
            close();
            return false;
        }

        bool compatible =
            (header_->num_tilings == num_tilings) &&
            (header_->grid_size_a == grid_size_a) &&
            (header_->grid_size_m == grid_size_m) &&
            (header_->memory_size == memory_size) &&
            (header_->num_actions == num_actions) &&
            (header_->num_way == num_way);

        if (!compatible) {
            std::cerr << "[SharedWeightTable] 超参数不匹配" << std::endl;
            std::cerr << "  文件: TILINGS=" << header_->num_tilings
                      << " GRID=" << header_->grid_size_a
                      << "x" << header_->grid_size_m
                      << " MEM=" << header_->memory_size
                      << " ACTIONS=" << header_->num_actions
                      << " WAY=" << header_->num_way << std::endl;
            std::cerr << "  期望: TILINGS=" << num_tilings
                      << " GRID=" << grid_size_a << "x" << grid_size_m
                      << " MEM=" << memory_size
                      << " ACTIONS=" << num_actions
                      << " WAY=" << num_way << std::endl;
            close();
            return false;
        }

        weights_ptr_ = reinterpret_cast<double*>(
            static_cast<char*>(mapped_) + sizeof(Header));
        total_weights_ = total_weights;

        std::cout << "[SharedWeightTable] 成功映射共享权重: " << path
                  << " (" << total_weights_ << " 权重, "
                  << expected_size << " 字节)" << std::endl;
        return true;
    }

    void close()
    {
        if (mapped_ && mapped_ != MAP_FAILED) {
            munmap(mapped_, mapped_size_);
            mapped_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        header_ = nullptr;
        weights_ptr_ = nullptr;
    }

    bool is_open() const { return weights_ptr_ != nullptr; }

    // ===== 文件锁操作（flock 级别，进程粒度）=====

    void read_lock() const
    {
        if (fd_ >= 0) flock(fd_, LOCK_SH);
    }

    void read_unlock() const
    {
        if (fd_ >= 0) flock(fd_, LOCK_UN);
    }

    void write_lock() const
    {
        if (fd_ >= 0) flock(fd_, LOCK_EX);
    }

    void write_unlock() const
    {
        if (fd_ >= 0) flock(fd_, LOCK_UN);
    }

    /**
     * @brief 将共享权重复制到本地向量（调用方需持有 read_lock）
     */
    void snapshot(std::vector<double>& local) const
    {
        if (!weights_ptr_) return;
        if (local.size() != total_weights_)
            local.resize(total_weights_);
        std::memcpy(local.data(), weights_ptr_,
                     total_weights_ * sizeof(double));
    }

    /**
     * @brief 将累积的增量应用到共享权重（调用方需持有 write_lock）
     */
    void apply_deltas(const std::vector<std::pair<size_t, double>>& deltas)
    {
        if (!weights_ptr_) return;
        for (const auto& d : deltas) {
            weights_ptr_[d.first] += d.second;
        }
    }

private:
    int fd_ = -1;
    void* mapped_ = nullptr;
    size_t mapped_size_ = 0;
    Header* header_ = nullptr;
    double* weights_ptr_ = nullptr;
    size_t total_weights_ = 0;
};

#endif // SHARED_WEIGHT_TABLE_H
