// RECREATED 2026-08-10 from session context (original file lost in workspace deletion).
// Path: src/Arena/arena_evaluation/arena_evaluation/arena_evaluation/processing/acoustics/solver.cpp
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;

// We need a custom label struct for Dijkstra
struct Label {
    float dist;
    int walls;
    int index;

    // For priority queue: min-heap based on total cost.
    // Total cost = 20 * log10(dist + mic_distance) + walls * wall_tl
    // Since we can't easily store the derived cost inside without adding parameters,
    // we just store the cost directly!
    float cost;

    bool operator>(const Label& other) const {
        return cost > other.cost;
    }
};

extern "C" {
    void solve_acoustic_field(
        const uint8_t* grid,
        int width,
        int height,
        float resolution,
        float start_x, // pixel coords
        float start_y, // pixel coords
        const float* target_xs,
        const float* target_ys,
        int num_targets,
        float wall_tl,
        float mic_distance,
        float* out_attenuations
    ) {
        // Initialize outputs
        for (int i = 0; i < num_targets; ++i) {
            out_attenuations[i] = numeric_limits<float>::infinity();
        }

        int start_ix = round(start_x);
        int start_iy = round(start_y);

        if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) {
            return;
        }

        // To do dominance pruning, we keep track of the minimum distance for each number of walls crossed at each cell.
        // We cap the maximum number of walls to 20 to bound the memory.
        const int MAX_WALLS = 20;

        // Memory size: width * height * MAX_WALLS * sizeof(float)
        // For 500x700, 350k cells * 20 * 4 = 28 MB. Easily fits in memory.
        vector<vector<float>> min_dist(width * height, vector<float>(MAX_WALLS, numeric_limits<float>::infinity()));

        priority_queue<Label, vector<Label>, greater<Label>> pq;

        // Initial state
        int start_idx = start_iy * width + start_ix;
        bool is_wall = grid[start_idx] > 0;
        int init_walls = is_wall ? 1 : 0;

        // To accurately reflect continuous coordinates, the start distance is 0.
        float init_cost = 20.0f * log10(0.0f + mic_distance) + init_walls * wall_tl;

        min_dist[start_idx][init_walls] = 0.0f;
        pq.push({0.0f, init_walls, start_idx, init_cost});

        // Convert targets to indices
        vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int tx = round(target_xs[i]);
            int ty = round(target_ys[i]);
            if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
                target_indices[i] = ty * width + tx;
            } else {
                target_indices[i] = -1;
            }
        }

        int targets_remaining = num_targets;

        // Directions (8-connected)
        int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
        float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

        while (!pq.empty()) {
            Label curr = pq.top();
            pq.pop();

            // If we found a strictly better path previously, skip
            if (curr.dist > min_dist[curr.index][curr.walls]) {
                continue;
            }

            int cx = curr.index % width;
            int cy = curr.index / width;

            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + dx[dir];
                int ny = cy + dy[dir];

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                int nidx = ny * width + nx;
                bool next_is_wall = grid[nidx] > 0;
                bool curr_is_wall = grid[curr.index] > 0;

                // We only add wall TL when ENTERING a wall from air (or starting in one).
                int nwalls = curr.walls;
                if (next_is_wall && !curr_is_wall) {
                    nwalls += 1;
                }

                if (nwalls >= MAX_WALLS) {
                    continue; // Prune
                }

                float ndist = curr.dist + step_dist[dir] * resolution;

                // Dominance check: if there is an existing path to nidx with <= ndist and <= nwalls, this new path is dominated.
                bool dominated = false;
                for (int w = 0; w <= nwalls; ++w) {
                    if (min_dist[nidx][w] <= ndist) {
                        dominated = true;
                        break;
                    }
                }

                if (!dominated) {
                    min_dist[nidx][nwalls] = ndist;
                    float ncost = 20.0f * log10(ndist + mic_distance) + nwalls * wall_tl;
                    pq.push({ndist, nwalls, nidx, ncost});
                }
            }
        }

        // Read out the best costs for targets
        for (int i = 0; i < num_targets; ++i) {
            int tidx = target_indices[i];
            if (tidx != -1) {
                float best_cost = numeric_limits<float>::infinity();
                for (int w = 0; w < MAX_WALLS; ++w) {
                    float dist = min_dist[tidx][w];
                    if (dist != numeric_limits<float>::infinity()) {
                        float cost = 20.0f * log10(dist + mic_distance) + w * wall_tl;
                        if (cost < best_cost) {
                            best_cost = cost;
                        }
                    }
                }
                out_attenuations[i] = best_cost;
            }
        }
    }
}
