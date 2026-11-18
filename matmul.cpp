#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct RunResult {
    double seconds;
    int64_t checksum;
};

// Умножение части строк матрицы A [start_row, end_row) на матрицу B
void MultiplyWorker(const std::vector<int64_t>& A,
    const std::vector<int64_t>& B,
    std::vector<int64_t>& C,
    int start_row,
    int end_row,
    int m,
    int k,
    std::atomic<int>& completed_rows) {
    for (int i = start_row; i < end_row; ++i) {
        const int row_offset_a = i * m;
        const int row_offset_c = i * k;

        for (int t = 0; t < m; ++t) {
            const int64_t a_val = A[row_offset_a + t];
            const int row_offset_b = t * k;

            for (int j = 0; j < k; ++j) {
                C[row_offset_c + j] += a_val * B[row_offset_b + j];
            }
        }
        completed_rows.fetch_add(1, std::memory_order_relaxed);
    }
}

// Поток отображения прогресса
void ProgressMonitor(const std::atomic<int>& completed_rows,
    int total_rows,
    std::atomic<bool>& done,
    bool show_progress) {
    if (!show_progress) return;
    int last_percent = -1;

    while (!done.load(std::memory_order_acquire)) {
        int current = completed_rows.load(std::memory_order_relaxed);
        int percent = (total_rows > 0) ? (current * 100 / total_rows) : 100;

        if (percent != last_percent && percent < 100) {
            std::cout << "\rПрогресс: " << percent << "%" << std::flush;
            last_percent = percent;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    std::cout << "\rПрогресс: 100%" << std::endl;
}

// Выполнение одного умножения матриц n x m и m x k на num_threads потоках
RunResult RunMultiplication(int n, int m, int k, int num_threads, bool show_progress, bool print_matrix) {
    std::vector<int64_t> A(static_cast<size_t>(n) * m);
    std::vector<int64_t> B(static_cast<size_t>(m) * k);
    std::vector<int64_t> C(static_cast<size_t>(n) * k, 0);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, 9);

    for (auto& val : A) val = dist(rng);
    for (auto& val : B) val = dist(rng);

    int active_threads = std::min(n, num_threads);
    std::atomic<int> completed_rows{ 0 };
    std::atomic<bool> done{ false };

    auto start_time = std::chrono::steady_clock::now();

    std::thread monitor_thread(ProgressMonitor,
        std::cref(completed_rows),
        n,
        std::ref(done),
        show_progress);

    std::vector<std::thread> workers;
    workers.reserve(active_threads);

    int base_rows = n / active_threads;
    int extra_rows = n % active_threads;
    int current_row = 0;

    for (int t = 0; t < active_threads; ++t) {
        int rows_for_thread = base_rows + (t < extra_rows ? 1 : 0);
        int end_row = current_row + rows_for_thread;

        workers.emplace_back(MultiplyWorker,
            std::cref(A),
            std::cref(B),
            std::ref(C),
            current_row,
            end_row,
            m,
            k,
            std::ref(completed_rows));

        current_row = end_row;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    done.store(true, std::memory_order_release);
    monitor_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    int64_t checksum = 0;
    for (int64_t val : C) {
        checksum += val;
    }

    if (print_matrix && n <= 10 && k <= 10) {
        std::cout << "Матрица C:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                std::cout << C[i * k + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    return { elapsed.count(), checksum };
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    // Режим автоматической серии замеров с сохранением в results.csv
    if (argc == 2 && std::string(argv[1]) == "--bench") {
        std::ofstream csv("results.csv");
        if (!csv.is_open()) {
            std::cerr << "Не удалось создать файл results.csv\n";
            return 1;
        }

        csv << "size;threads;run;time_seconds;checksum\n";
        std::vector<int> sizes = { 550, 1200, 2500 };
        std::vector<int> threads = { 1, 2, 3, 4, 5, 6, 8, 12, 16 };

        for (int n : sizes) {
            for (int t : threads) {
                std::cout << "Running N=" << n << ", T=" << t << "...\n";
                for (int run = 1; run <= 5; ++run) {
                    RunResult res = RunMultiplication(n, n, n, t, false, false);
                    csv << n << ";" << t << ";" << run << ";" << res.seconds << ";" << res.checksum << "\n";
                    csv.flush();
                }
            }
        }
        std::cout << "Готово! Результаты сохранены в results.csv\n";
        return 0;
    }

    // Обычный режим одиночного запуска: n m k T
    int n = 0, m = 0, k = 0, num_threads = 0;
    if (argc == 5) {
        n = std::stoi(argv[1]);
        m = std::stoi(argv[2]);
        k = std::stoi(argv[3]);
        num_threads = std::stoi(argv[4]);
    }
    else {
        std::cout << "Введите n, m, k и количество потоков T: ";
        if (!(std::cin >> n >> m >> k >> num_threads)) {
            std::cerr << "Ошибка ввода параметров.\n";
            return 1;
        }
    }

    if (n <= 0 || m <= 0 || k <= 0 || num_threads < 1) {
        std::cerr << "Размеры матриц должны быть > 0, а число потоков >= 1.\n";
        return 1;
    }

    RunResult res = RunMultiplication(n, m, k, num_threads, true, true);
    std::cout << "Время умножения: " << res.seconds << " сек.\n";
    std::cout << "Контрольная сумма: " << res.checksum << "\n";

    return 0;
}