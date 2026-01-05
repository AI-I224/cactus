#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <arm_neon.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unistd.h>
#include <unordered_map>
#include <chrono>
#include <string>
#include <cstdio>

constexpr size_t NEON_VECTOR_SIZE = 16;

inline int8_t clamp_to_int8(float value) {
    int32_t clamped = static_cast<int32_t>(roundf(value));
    return static_cast<int8_t>(std::max(-128, std::min(127, clamped)));
}

inline int8_t clamp_to_int8(int32_t value) {
    return static_cast<int8_t>(std::max(-128, std::min(127, value)));
}

#if defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t accum_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
inline int32x4_t accum_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int16x8_t prod_low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int32x4_t acc_high = vpaddlq_s16(vmull_s8(vget_high_s8(a), vget_high_s8(b)));
    return vaddq_s32(vaddq_s32(acc, vpaddlq_s16(prod_low)), acc_high);
}
#endif

inline float16x8_t accum_f16_dot(float16x8_t acc, float16x8_t a_low, float16x8_t a_high, 
                                 float16x8_t b_low, float16x8_t b_high) {
    acc = vfmaq_f16(acc, a_low, b_low);
    return vfmaq_f16(acc, a_high, b_high);
}

inline float32x4_t accum_f32_dot(float32x4_t acc, float32x4_t a_low, float32x4_t a_high, 
                                  float32x4_t b_low, float32x4_t b_high) {
    acc = vfmaq_f32(acc, a_low, b_low);
    return vfmaq_f32(acc, a_high, b_high);
}

namespace CactusThreading {

    class ThreadPool {
    private:
        static constexpr size_t MAX_WORKERS = 16;

        struct WorkerQueue {
            std::deque<std::function<void()>> tasks;
            std::mutex mutex;
        };

        std::vector<std::thread> workers;
        std::vector<std::unique_ptr<WorkerQueue>> worker_queues;
        size_t num_workers_;

        std::atomic<bool> stop{false};
        std::atomic<size_t> active_workers{0};
        std::atomic<size_t> total_tasks{0};
        std::mutex wait_mutex;
        std::condition_variable wait_condition;

        bool try_steal(size_t thief_id, std::function<void()>& task) {
            for (size_t i = 1; i < num_workers_; ++i) {
                size_t victim = (thief_id + i) % num_workers_;
                auto& victim_queue = *worker_queues[victim];

                std::unique_lock<std::mutex> lock(victim_queue.mutex, std::try_to_lock);
                if (lock.owns_lock() && !victim_queue.tasks.empty()) {
                    task = std::move(victim_queue.tasks.front());
                    victim_queue.tasks.pop_front();
                    return true;
                }
            }
            return false;
        }

        void worker_thread(size_t id) {
            auto& my_queue = *worker_queues[id];

            while (!stop.load(std::memory_order_relaxed)) {
                std::function<void()> task;
                bool got_task = false;

                {
                    std::lock_guard<std::mutex> lock(my_queue.mutex);
                    if (!my_queue.tasks.empty()) {
                        task = std::move(my_queue.tasks.back());
                        my_queue.tasks.pop_back();
                        got_task = true;
                    }
                }

                if (!got_task) {
                    got_task = try_steal(id, task);
                }

                if (got_task) {
                    active_workers.fetch_add(1, std::memory_order_relaxed);
                    task();
                    active_workers.fetch_sub(1, std::memory_order_relaxed);

                    if (total_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        wait_condition.notify_all();
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
            num_workers_ = std::min(num_threads, MAX_WORKERS);

            worker_queues.reserve(num_workers_);
            for (size_t i = 0; i < num_workers_; ++i) {
                worker_queues.push_back(std::make_unique<WorkerQueue>());
            }

            workers.reserve(num_workers_);
            for (size_t i = 0; i < num_workers_; ++i) {
                workers.emplace_back(&ThreadPool::worker_thread, this, i);
            }
        }

        ~ThreadPool() {
            stop.store(true, std::memory_order_release);
            for (auto& worker : workers) {
                worker.join();
            }
        }

        template<typename F>
        auto enqueue(F&& f) -> std::future<decltype(f())> {
            using return_type = decltype(f());

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::forward<F>(f)
            );

            std::future<return_type> res = task->get_future();

            size_t target = total_tasks.fetch_add(1, std::memory_order_relaxed) % num_workers_;
            {
                std::lock_guard<std::mutex> lock(worker_queues[target]->mutex);
                worker_queues[target]->tasks.emplace_back([task](){ (*task)(); });
            }

            return res;
        }

        template<typename F>
        void enqueue_batch(size_t total_work, F task_func) {
            if (total_work == 0) return;

            const size_t num_tasks = std::min(num_workers_, total_work);
            const size_t per_worker = total_work / num_tasks;
            const size_t remainder = total_work % num_tasks;

            total_tasks.fetch_add(num_tasks, std::memory_order_relaxed);

            for (size_t w = 0; w < num_tasks; ++w) {
                size_t start = w * per_worker + std::min(w, remainder);
                size_t end = start + per_worker + (w < remainder ? 1 : 0);

                std::lock_guard<std::mutex> lock(worker_queues[w]->mutex);
                worker_queues[w]->tasks.emplace_back(
                    [=]() { task_func(start, end); }
                );
            }
        }

        void wait_all() {
            std::unique_lock<std::mutex> lock(wait_mutex);
            wait_condition.wait(lock, [this] {
                return total_tasks.load(std::memory_order_acquire) == 0;
            });
        }

        template<typename F>
        void enqueue_n_threads(size_t total_work, size_t num_threads, F task_func) {
            if (total_work == 0 || num_threads == 0) return;

            num_threads = std::min(num_threads, std::min(num_workers_, total_work));
            const size_t per_thread = total_work / num_threads;
            const size_t remainder = total_work % num_threads;

            total_tasks.fetch_add(num_threads, std::memory_order_relaxed);

            for (size_t t = 0; t < num_threads; ++t) {
                size_t start = t * per_thread + std::min(t, remainder);
                size_t end = start + per_thread + (t < remainder ? 1 : 0);

                std::lock_guard<std::mutex> lock(worker_queues[t % num_workers_]->mutex);
                worker_queues[t % num_workers_]->tasks.emplace_back(
                    [=]() { task_func(start, end); }
                );
            }
        }

        size_t num_workers() const { return num_workers_; }
    };

    inline ThreadPool& get_thread_pool() {
        static ThreadPool pool;
        return pool;
    }
    
    struct ParallelConfig {
        size_t min_work_gate;  
        size_t work_per_thread; 

        constexpr ParallelConfig(size_t gate, size_t per_thread)
            : min_work_gate(gate), work_per_thread(per_thread) {}
    };

    inline size_t get_optimal_thread_count(size_t total_work, ParallelConfig config) {
        if (total_work < config.min_work_gate) return 1;

        size_t pool_size = get_thread_pool().num_workers();
        size_t num_threads = (total_work + config.work_per_thread - 1) / config.work_per_thread;
        return std::min(pool_size, std::max(static_cast<size_t>(1), num_threads));
    }

    struct Thresholds {
        #if defined(__ANDROID__)
        static constexpr ParallelConfig ATTENTION{64, 32};
        static constexpr ParallelConfig ELEMENT_WISE{5000, 2500};
        static constexpr ParallelConfig AXIS_REDUCE{1000, 500};
        static constexpr ParallelConfig ALL_REDUCE{10000, 5000};
        static constexpr ParallelConfig SCALAR_BASIC{30000, 15000};
        static constexpr ParallelConfig SCALAR_EXPENSIVE{10000, 5000};
        #else // Apple
        static constexpr ParallelConfig ATTENTION{32, 16};
        static constexpr ParallelConfig ELEMENT_WISE{5000, 2500};
        static constexpr ParallelConfig AXIS_REDUCE{1000, 500};
        static constexpr ParallelConfig ALL_REDUCE{10000, 5000};
        static constexpr ParallelConfig SCALAR_BASIC{5000, 2500};
        static constexpr ParallelConfig SCALAR_EXPENSIVE{2500, 1250};
        #endif
    };

    struct GemmThreading {
        #if defined(__ANDROID__)
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return 1; 
            return pool_size; 
        }
        #elif defined(__APPLE__) && TARGET_OS_IPHONE
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(2)); 
            return pool_size; 
        }
        #else // Mac
        static size_t get_num_threads(size_t M, size_t pool_size) {
            if (M <= 1) return std::min(pool_size, static_cast<size_t>(4));
            return pool_size; 
        }
        #endif
    };

    inline size_t& get_gemm_thread_override() {
        static size_t override_threads = 0; 
        return override_threads;
    }

    inline void set_gemm_threads(size_t num_threads) {
        get_gemm_thread_override() = num_threads;
    }

    inline void reset_gemm_threads() {
        get_gemm_thread_override() = 0;
    }
    
    class TaskHandle {
    private:
        std::vector<std::future<void>> futures_;
        bool auto_wait_;
        
    public:
        TaskHandle(bool auto_wait = true) : auto_wait_(auto_wait) {}
        
        ~TaskHandle() {
            if (auto_wait_) {
                wait();
            }
        }
        
        TaskHandle(TaskHandle&&) = default;
        TaskHandle& operator=(TaskHandle&&) = default;
        TaskHandle(const TaskHandle&) = delete;
        TaskHandle& operator=(const TaskHandle&) = delete;
        
        void add_future(std::future<void>&& f) {
            futures_.push_back(std::move(f));
        }
        
        void wait() {
            for (auto& f : futures_) {
                if (f.valid()) {
                    f.wait();
                }
            }
            futures_.clear();
        }
        
        bool is_ready() const {
            for (const auto& f : futures_) {
                if (f.valid() && f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    return false;
                }
            }
            return true;
        }
        
        size_t task_count() const { return futures_.size(); }
    };
    
    template<typename WorkFunc>
    TaskHandle parallel_for(size_t total_work, ParallelConfig config, WorkFunc work_func, bool wait = true) {
        const size_t num_threads = get_optimal_thread_count(total_work, config);
        TaskHandle handle(!wait);

        if (num_threads == 1) {
            if (wait) {
                work_func(0, total_work);
                return handle;
            }
            auto& pool = get_thread_pool();
            handle.add_future(pool.enqueue([work_func, total_work]() {
                work_func(0, total_work);
            }));
            return handle;
        }

        auto& pool = get_thread_pool();
        const size_t work_per_thread = total_work / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            handle.add_future(pool.enqueue([work_func, t, num_threads, work_per_thread, total_work]() {
                const size_t start_idx = t * work_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? total_work : (t + 1) * work_per_thread;
                work_func(start_idx, end_idx);
            }));
        }

        if (wait) {
            handle.wait();
        }
        return handle;
    }

    template<typename WorkFunc>
    void parallel_for_2d(size_t outer_size, size_t inner_size, ParallelConfig config, WorkFunc work_func) {
        const size_t total_work = outer_size * inner_size;
        parallel_for(total_work, config, [&](size_t start_idx, size_t end_idx) {
            for (size_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
                const size_t outer = work_idx / inner_size;
                const size_t inner = work_idx % inner_size;
                work_func(outer, inner);
            }
        });
    }

    template<typename WorkFunc, typename ResultType, typename CombineFunc>
    ResultType parallel_reduce(size_t total_work, ParallelConfig config,
                              WorkFunc work_func, ResultType init_value, CombineFunc combine_func) {
        const size_t num_threads = get_optimal_thread_count(total_work, config);
        
        if (num_threads == 1) {
            return work_func(0, total_work);
        }
        
        auto& pool = get_thread_pool();
        std::vector<std::future<ResultType>> futures;
        std::vector<ResultType> partial_results(num_threads, init_value);
        const size_t work_per_thread = total_work / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            futures.push_back(pool.enqueue([work_func, t, num_threads, work_per_thread, total_work]() -> ResultType {
                const size_t start_idx = t * work_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? total_work : (t + 1) * work_per_thread;
                return work_func(start_idx, end_idx);
            }));
        }
        
        ResultType result = init_value;
        for (auto& future : futures) {
            result = combine_func(result, future.get());
        }
        return result;
    }

    template<typename WorkFunc>
    void parallel_for_2d_tiled_gemm(size_t M, size_t rows, size_t cols, size_t tile_rows, size_t tile_cols, WorkFunc work_func) {
        size_t num_row_tiles = (rows + tile_rows - 1) / tile_rows;
        size_t num_col_tiles = (cols + tile_cols - 1) / tile_cols;
        size_t total_tiles = num_row_tiles * num_col_tiles;

        auto& pool = get_thread_pool();

        size_t override = get_gemm_thread_override();
        size_t num_threads = (override > 0) ? override : GemmThreading::get_num_threads(M, pool.num_workers());
        num_threads = std::min(num_threads, total_tiles);

        if (num_threads <= 1) {
            for (size_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
                size_t tile_row = tile_idx / num_col_tiles;
                size_t tile_col = tile_idx % num_col_tiles;
                size_t row_start = tile_row * tile_rows;
                size_t row_end = std::min(row_start + tile_rows, rows);
                size_t col_start = tile_col * tile_cols;
                size_t col_end = std::min(col_start + tile_cols, cols);
                work_func(row_start, row_end, col_start, col_end);
            }
            return;
        }

        pool.enqueue_n_threads(total_tiles, num_threads,
            [=](size_t start_tile, size_t end_tile) {
                for (size_t tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
                    size_t tile_row = tile_idx / num_col_tiles;
                    size_t tile_col = tile_idx % num_col_tiles;
                    size_t row_start = tile_row * tile_rows;
                    size_t row_end = std::min(row_start + tile_rows, rows);
                    size_t col_start = tile_col * tile_cols;
                    size_t col_end = std::min(col_start + tile_cols, cols);
                    work_func(row_start, row_end, col_start, col_end);
                }
            });
        pool.wait_all();
    }

    template<typename WorkFunc>
    void parallel_for_2d_tiled(size_t rows, size_t cols, size_t tile_rows, size_t tile_cols, ParallelConfig config, WorkFunc work_func) {
        size_t num_row_tiles = (rows + tile_rows - 1) / tile_rows;
        size_t num_col_tiles = (cols + tile_cols - 1) / tile_cols;
        size_t total_tiles = num_row_tiles * num_col_tiles;

        if (total_tiles < config.min_work_gate) {
            for (size_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
                size_t tile_row = tile_idx / num_col_tiles;
                size_t tile_col = tile_idx % num_col_tiles;
                size_t row_start = tile_row * tile_rows;
                size_t row_end = std::min(row_start + tile_rows, rows);
                size_t col_start = tile_col * tile_cols;
                size_t col_end = std::min(col_start + tile_cols, cols);
                work_func(row_start, row_end, col_start, col_end);
            }
            return;
        }

        auto& pool = get_thread_pool();
        pool.enqueue_batch(total_tiles,
            [=](size_t start_tile, size_t end_tile) {
                for (size_t tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
                    size_t tile_row = tile_idx / num_col_tiles;
                    size_t tile_col = tile_idx % num_col_tiles;
                    size_t row_start = tile_row * tile_rows;
                    size_t row_end = std::min(row_start + tile_rows, rows);
                    size_t col_start = tile_col * tile_cols;
                    size_t col_end = std::min(col_start + tile_cols, cols);
                    work_func(row_start, row_end, col_start, col_end);
                }
            });
        pool.wait_all();
    }
}


#endif // KERNEL_UTILS_H 