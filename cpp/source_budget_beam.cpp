#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using std::cerr;
using std::cout;
using std::int32_t;
using std::size_t;
using std::string;
using std::uint16_t;
using std::vector;

struct Config {
    string data_dir;
    string out_dir;
    double target_avg = 800.0;
    double hard_avg_limit = 850.0;
    int p95_limit = 1200;
    int per_bucket = 4;
    int final_top_n = 20;
    int time_limit_sec = 0;
    int progress_interval_sec = 60;
    double lambda_over = 1.0;
    vector<double> bucket_bounds = {0, 300, 500, 700, 800, 850, 900, 1100, 1e18};
};

struct Meta {
    int num_turns = 0;
    int num_sources = 0;
    int num_tracks = 0;
    int max_k = 0;
    vector<int> k_values;
    string candidate_file = "candidates.i32";
    string counts_file = "counts.u16";
    string gold_file = "gold.i32";
    string sources_file = "sources.tsv";
};

struct SourceInfo {
    int index = 0;
    string name;
    string index_name;
    string variant_name;
};

struct State {
    vector<uint16_t> choices;
    std::shared_ptr<vector<vector<int>>> unions_by_turn;
    vector<uint16_t> sizes;
    int hit = 0;
    long long total_size = 0;
    int p95 = 0;
    int max_size = 0;
    double avg_size = 0.0;
};

static vector<string> split_ws(const string& line) {
    std::istringstream iss(line);
    vector<string> parts;
    string item;
    while (iss >> item) parts.push_back(item);
    return parts;
}

static vector<string> split_tab(const string& line) {
    vector<string> parts;
    string cur;
    for (char ch : line) {
        if (ch == '\t') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    parts.push_back(cur);
    return parts;
}

static string join_path(const string& a, const string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static bool starts_with(const string& s, const string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto require_value = [&](const string& name) -> string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (arg == "--data-dir") cfg.data_dir = require_value(arg);
        else if (arg == "--out-dir") cfg.out_dir = require_value(arg);
        else if (arg == "--target-avg") cfg.target_avg = std::stod(require_value(arg));
        else if (arg == "--hard-avg-limit") cfg.hard_avg_limit = std::stod(require_value(arg));
        else if (arg == "--p95-limit") cfg.p95_limit = std::stoi(require_value(arg));
        else if (arg == "--per-bucket") cfg.per_bucket = std::stoi(require_value(arg));
        else if (arg == "--final-top-n") cfg.final_top_n = std::stoi(require_value(arg));
        else if (arg == "--time-limit-sec") cfg.time_limit_sec = std::stoi(require_value(arg));
        else if (arg == "--progress-interval-sec") cfg.progress_interval_sec = std::stoi(require_value(arg));
        else if (arg == "--lambda-over") cfg.lambda_over = std::stod(require_value(arg));
        else if (arg == "--bucket-bounds") {
            cfg.bucket_bounds.clear();
            for (const auto& part : split_ws(require_value(arg))) cfg.bucket_bounds.push_back(std::stod(part));
            if (cfg.bucket_bounds.empty() || cfg.bucket_bounds.front() != 0) cfg.bucket_bounds.insert(cfg.bucket_bounds.begin(), 0);
            cfg.bucket_bounds.push_back(1e18);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (cfg.data_dir.empty()) throw std::runtime_error("--data-dir is required");
    if (cfg.out_dir.empty()) cfg.out_dir = cfg.data_dir + "/beam_search";
    if (cfg.per_bucket <= 0) throw std::runtime_error("--per-bucket must be positive");
    return cfg;
}

static Meta read_meta(const string& data_dir) {
    Meta meta;
    std::ifstream in(join_path(data_dir, "meta.txt"));
    if (!in) throw std::runtime_error("Could not open meta.txt");
    string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto parts = split_tab(line);
        if (parts.size() < 2) continue;
        const string& key = parts[0];
        const string& value = parts[1];
        if (key == "num_turns") meta.num_turns = std::stoi(value);
        else if (key == "num_sources") meta.num_sources = std::stoi(value);
        else if (key == "num_tracks") meta.num_tracks = std::stoi(value);
        else if (key == "max_k") meta.max_k = std::stoi(value);
        else if (key == "k_values") {
            for (const auto& item : split_ws(value)) meta.k_values.push_back(std::stoi(item));
        } else if (key == "candidate_file") meta.candidate_file = value;
        else if (key == "counts_file") meta.counts_file = value;
        else if (key == "gold_file") meta.gold_file = value;
        else if (key == "sources_file") meta.sources_file = value;
    }
    if (meta.num_turns <= 0 || meta.num_sources <= 0 || meta.max_k <= 0 || meta.num_tracks <= 0) {
        throw std::runtime_error("Invalid meta.txt dimensions");
    }
    if (meta.k_values.empty()) meta.k_values = {0, 50, 100, 200, 400, 800};
    if (meta.k_values.front() != 0) meta.k_values.insert(meta.k_values.begin(), 0);
    return meta;
}

template <class T>
static vector<T> read_binary_vector(const string& path, size_t expected) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open binary file: " + path);
    vector<T> data(expected);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected * sizeof(T)));
    if (static_cast<size_t>(in.gcount()) != expected * sizeof(T)) {
        throw std::runtime_error("Unexpected byte count while reading: " + path);
    }
    return data;
}

static vector<SourceInfo> read_sources(const string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open sources.tsv");
    vector<SourceInfo> sources;
    string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        auto parts = split_tab(line);
        if (parts.size() < 4) continue;
        SourceInfo info;
        info.index = std::stoi(parts[0]);
        info.name = parts[1];
        info.index_name = parts[2];
        info.variant_name = parts[3];
        sources.push_back(info);
    }
    return sources;
}

static inline size_t offset_for(int t, int s, int max_k, int num_sources) {
    return (static_cast<size_t>(t) * num_sources + s) * max_k;
}

static int compute_p95(vector<uint16_t> sizes) {
    if (sizes.empty()) return 0;
    size_t pos = static_cast<size_t>(std::ceil(0.95 * sizes.size()));
    if (pos == 0) pos = 1;
    pos -= 1;
    std::nth_element(sizes.begin(), sizes.begin() + static_cast<long>(pos), sizes.end());
    return sizes[pos];
}

static double score_state(const State& state, const Config& cfg) {
    double score = state.hit;
    if (state.avg_size > cfg.target_avg) score -= cfg.lambda_over * (state.avg_size - cfg.target_avg);
    if (cfg.p95_limit > 0 && state.p95 > cfg.p95_limit) score -= 10.0 * (state.p95 - cfg.p95_limit);
    return score;
}

static bool better_state(const std::shared_ptr<State>& a, const std::shared_ptr<State>& b, const Config& cfg) {
    double sa = score_state(*a, cfg);
    double sb = score_state(*b, cfg);
    if (std::abs(sa - sb) > 1e-9) return sa > sb;
    if (a->hit != b->hit) return a->hit > b->hit;
    double da = std::abs(a->avg_size - cfg.target_avg);
    double db = std::abs(b->avg_size - cfg.target_avg);
    if (std::abs(da - db) > 1e-9) return da < db;
    if (a->p95 != b->p95) return a->p95 < b->p95;
    return a->max_size < b->max_size;
}

static int bucket_for(double avg, const vector<double>& bounds) {
    for (int i = 0; i + 1 < static_cast<int>(bounds.size()); ++i) {
        if (avg >= bounds[i] && avg < bounds[i + 1]) return i;
    }
    return static_cast<int>(bounds.size()) - 2;
}

static bool feasible_for_prune(const State& state, const Config& cfg) {
    if (cfg.hard_avg_limit > 0 && state.avg_size > cfg.hard_avg_limit) return false;
    if (cfg.p95_limit > 0 && state.p95 > cfg.p95_limit) return false;
    return true;
}

static void insert_bucketed(
    vector<vector<std::shared_ptr<State>>>& buckets,
    std::shared_ptr<State> state,
    const Config& cfg
) {
    if (!feasible_for_prune(*state, cfg)) return;
    int bucket = bucket_for(state->avg_size, cfg.bucket_bounds);
    auto& vec = buckets[bucket];
    vec.push_back(std::move(state));
    std::sort(vec.begin(), vec.end(), [&](const auto& lhs, const auto& rhs) {
        return better_state(lhs, rhs, cfg);
    });
    if (static_cast<int>(vec.size()) > cfg.per_bucket) vec.pop_back();
}

static std::shared_ptr<State> current_best_from_buckets(
    const vector<vector<std::shared_ptr<State>>>& buckets,
    const Config& cfg
) {
    std::shared_ptr<State> best;
    for (const auto& bucket : buckets) {
        for (const auto& state : bucket) {
            if (!best || better_state(state, best, cfg)) best = state;
        }
    }
    return best;
}

static std::shared_ptr<State> add_source(
    const State& parent,
    int source,
    int selected_k,
    const Meta& meta,
    const vector<int32_t>& candidates,
    const vector<uint16_t>& counts,
    const vector<int32_t>& gold
) {
    auto child = std::make_shared<State>();
    child->choices = parent.choices;
    child->choices[source] = static_cast<uint16_t>(selected_k);
    if (selected_k <= 0) {
        child->unions_by_turn = parent.unions_by_turn;
        child->sizes = parent.sizes;
        child->hit = parent.hit;
        child->total_size = parent.total_size;
        child->p95 = parent.p95;
        child->max_size = parent.max_size;
        child->avg_size = parent.avg_size;
        return child;
    }

    auto unions = std::make_shared<vector<vector<int>>>();
    unions->resize(meta.num_turns);
    child->sizes.assign(meta.num_turns, 0);

    vector<int> temp;
    vector<int> merged;
    long long total_size = 0;
    int hit = 0;
    int max_size = 0;

    for (int t = 0; t < meta.num_turns; ++t) {
        const auto& base = (*(parent.unions_by_turn))[t];
        int count = counts[static_cast<size_t>(t) * meta.num_sources + source];
        int take = std::min(selected_k, count);
        temp.clear();
        temp.reserve(take);
        size_t offset = offset_for(t, source, meta.max_k, meta.num_sources);
        for (int i = 0; i < take; ++i) {
            int id = candidates[offset + i];
            if (id >= 0) temp.push_back(id);
        }
        std::sort(temp.begin(), temp.end());
        temp.erase(std::unique(temp.begin(), temp.end()), temp.end());

        merged.clear();
        merged.reserve(base.size() + temp.size());
        std::merge(base.begin(), base.end(), temp.begin(), temp.end(), std::back_inserter(merged));
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
        int size = static_cast<int>(merged.size());
        child->sizes[t] = static_cast<uint16_t>(std::min(size, 65535));
        total_size += size;
        max_size = std::max(max_size, size);
        int gold_id = gold[t];
        if (gold_id >= 0 && std::binary_search(merged.begin(), merged.end(), gold_id)) ++hit;
        (*unions)[t] = merged;
    }

    child->unions_by_turn = std::move(unions);
    child->hit = hit;
    child->total_size = total_size;
    child->avg_size = static_cast<double>(total_size) / meta.num_turns;
    child->p95 = compute_p95(child->sizes);
    child->max_size = max_size;
    return child;
}

static vector<int> source_order_by_single_hit(
    const Meta& meta,
    const vector<int32_t>& candidates,
    const vector<uint16_t>& counts,
    const vector<int32_t>& gold
) {
    vector<std::pair<int, int>> scored;
    int max_choice = 0;
    for (int k : meta.k_values) max_choice = std::max(max_choice, k);
    for (int s = 0; s < meta.num_sources; ++s) {
        int hit = 0;
        for (int t = 0; t < meta.num_turns; ++t) {
            int take = std::min<int>(max_choice, counts[static_cast<size_t>(t) * meta.num_sources + s]);
            size_t offset = offset_for(t, s, meta.max_k, meta.num_sources);
            int gold_id = gold[t];
            bool found = false;
            for (int i = 0; i < take; ++i) {
                if (candidates[offset + i] == gold_id) {
                    found = true;
                    break;
                }
            }
            if (found) ++hit;
        }
        scored.push_back({-hit, s});
    }
    std::sort(scored.begin(), scored.end());
    vector<int> order;
    for (auto [neg_hit, source] : scored) order.push_back(source);
    return order;
}

static string choices_to_string(const State& state, const vector<SourceInfo>& sources) {
    std::ostringstream oss;
    bool first = true;
    for (int s = 0; s < static_cast<int>(state.choices.size()); ++s) {
        if (state.choices[s] == 0) continue;
        if (!first) oss << "|";
        first = false;
        oss << sources[s].name << "=" << state.choices[s];
    }
    return oss.str();
}

static void ensure_out_dir(const string& out_dir) {
    string cmd = "mkdir -p '" + out_dir + "'";
    int rc = std::system(cmd.c_str());
    if (rc != 0) throw std::runtime_error("Failed to create output directory: " + out_dir);
}

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        ensure_out_dir(cfg.out_dir);
        Meta meta = read_meta(cfg.data_dir);
        auto sources = read_sources(join_path(cfg.data_dir, meta.sources_file));
        if (static_cast<int>(sources.size()) != meta.num_sources) {
            throw std::runtime_error("sources.tsv count does not match meta.num_sources");
        }

        size_t candidate_count = static_cast<size_t>(meta.num_turns) * meta.num_sources * meta.max_k;
        size_t source_turn_count = static_cast<size_t>(meta.num_turns) * meta.num_sources;
        auto candidates = read_binary_vector<int32_t>(join_path(cfg.data_dir, meta.candidate_file), candidate_count);
        auto counts = read_binary_vector<uint16_t>(join_path(cfg.data_dir, meta.counts_file), source_turn_count);
        auto gold = read_binary_vector<int32_t>(join_path(cfg.data_dir, meta.gold_file), meta.num_turns);

        auto initial = std::make_shared<State>();
        initial->choices.assign(meta.num_sources, 0);
        initial->unions_by_turn = std::make_shared<vector<vector<int>>>(meta.num_turns);
        initial->sizes.assign(meta.num_turns, 0);
        vector<std::shared_ptr<State>> beam = {initial};
        vector<int> order = source_order_by_single_hit(meta, candidates, counts, gold);
        auto started_at = std::chrono::steady_clock::now();
        auto last_progress_at = started_at;
        bool stopped_by_time_limit = false;

        std::ofstream progress(join_path(cfg.out_dir, "beam_progress.tsv"));
        progress << "step\tsource_index\tsource_name\tbeam_states\tbest_hit\tbest_coverage\tbest_avg_size\tbest_p95\n";

        for (int step = 0; step < static_cast<int>(order.size()); ++step) {
            int source = order[step];
            vector<vector<std::shared_ptr<State>>> buckets(cfg.bucket_bounds.size() - 1);
            for (int parent_index = 0; parent_index < static_cast<int>(beam.size()); ++parent_index) {
                const auto& parent = beam[parent_index];
                for (int k : meta.k_values) {
                    if (k > meta.max_k) continue;
                    auto child = add_source(*parent, source, k, meta, candidates, counts, gold);
                    insert_bucketed(buckets, std::move(child), cfg);
                }
                auto now = std::chrono::steady_clock::now();
                int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - started_at).count());
                int since_progress = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_at).count());
                if (cfg.progress_interval_sec > 0 && since_progress >= cfg.progress_interval_sec) {
                    auto best_partial = current_best_from_buckets(buckets, cfg);
                    if (best_partial) {
                        cout << "progress elapsed_sec=" << elapsed
                             << " step=" << (step + 1) << "/" << order.size()
                             << " source=" << sources[source].name
                             << " parents_done=" << (parent_index + 1) << "/" << beam.size()
                             << " partial_best_hit=" << best_partial->hit
                             << " partial_coverage=" << (static_cast<double>(best_partial->hit) / meta.num_turns)
                             << " partial_avg=" << best_partial->avg_size
                             << " partial_p95=" << best_partial->p95 << "\n";
                    } else {
                        cout << "progress elapsed_sec=" << elapsed
                             << " step=" << (step + 1) << "/" << order.size()
                             << " source=" << sources[source].name
                             << " parents_done=" << (parent_index + 1) << "/" << beam.size()
                             << " partial_best_hit=NA\n";
                    }
                    last_progress_at = now;
                }
            }
            beam.clear();
            for (auto& bucket : buckets) {
                for (auto& state : bucket) beam.push_back(std::move(state));
            }
            std::sort(beam.begin(), beam.end(), [&](const auto& lhs, const auto& rhs) {
                return better_state(lhs, rhs, cfg);
            });

            if (beam.empty()) {
                cerr << "Beam became empty at source " << source << "\n";
                return 2;
            }
            const auto& best = beam.front();
            progress << (step + 1) << "\t" << source << "\t" << sources[source].name << "\t" << beam.size()
                     << "\t" << best->hit << "\t" << (static_cast<double>(best->hit) / meta.num_turns)
                     << "\t" << best->avg_size << "\t" << best->p95 << "\n";
            cout << "step " << (step + 1) << "/" << order.size() << " source=" << sources[source].name
                 << " beam=" << beam.size() << " best_hit=" << best->hit
                 << " coverage=" << (static_cast<double>(best->hit) / meta.num_turns)
                 << " avg=" << best->avg_size << " p95=" << best->p95 << "\n";
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - started_at).count());
            if (cfg.time_limit_sec > 0 && elapsed >= cfg.time_limit_sec && step + 1 < static_cast<int>(order.size())) {
                cout << "time_limit_reached elapsed_sec=" << elapsed
                     << " after_step=" << (step + 1)
                     << " writing best states from partial search\n";
                stopped_by_time_limit = true;
                break;
            }
        }

        std::sort(beam.begin(), beam.end(), [&](const auto& lhs, const auto& rhs) {
            return better_state(lhs, rhs, cfg);
        });

        std::ofstream results(join_path(cfg.out_dir, "best_states.csv"));
        results << "rank,hit,coverage,avg_union_size,p95_union_size,max_union_size,score,choices\n";
        int limit = std::min<int>(cfg.final_top_n, beam.size());
        for (int i = 0; i < limit; ++i) {
            const auto& state = beam[i];
            results << (i + 1) << "," << state->hit << "," << (static_cast<double>(state->hit) / meta.num_turns)
                    << "," << state->avg_size << "," << state->p95 << "," << state->max_size << ","
                    << score_state(*state, cfg) << ",\"" << choices_to_string(*state, sources) << "\"\n";
        }

        const auto& best = beam.front();
        std::ofstream choice(join_path(cfg.out_dir, "best_choice.tsv"));
        choice << "source_index\tsource_name\tselected_k\n";
        for (int s = 0; s < meta.num_sources; ++s) {
            if (best->choices[s] == 0) continue;
            choice << s << "\t" << sources[s].name << "\t" << best->choices[s] << "\n";
        }

        cout << "best_hit=" << best->hit << " coverage=" << (static_cast<double>(best->hit) / meta.num_turns)
             << " avg_union_size=" << best->avg_size << " p95=" << best->p95
             << " stopped_by_time_limit=" << (stopped_by_time_limit ? "true" : "false")
             << " out_dir=" << cfg.out_dir << "\n";
    } catch (const std::exception& ex) {
        cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
