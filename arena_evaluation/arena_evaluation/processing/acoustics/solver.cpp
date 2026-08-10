// solver v2: per-pixel transmission-loss map (optional).
// v1 semantics preserved when pixel_tl == NULL: crossing into a wall pixel
// (air -> wall transition) costs wall_tl once per wall crossing, regardless
// of wall thickness. With pixel_tl, the transition costs pixel_tl[p] of the
// entered pixel (walls 47, closed doors 25, glass 20, free 0); an OPEN door
// has TL 0 so it behaves as free space. The accumulated TL rides in the label
// so exiting the barrier does not lose it; min_cost per cell is the
// authoritative readout.
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;

struct Label {
    float dist;
    int walls;
    int index;
    float tl;   // accumulated transmission loss along the path
    float cost; // 20*log10(dist + mic) + tl

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
        float start_x,
        float start_y,
        const float* target_xs,
        const float* target_ys,
        int num_targets,
        float wall_tl,
        float mic_distance,
        const float* pixel_tl, // optional row-major TL map; NULL => wall_tl per wall crossing
        float* out_attenuations
    ) {
        for (int i = 0; i < num_targets; ++i) {
            out_attenuations[i] = numeric_limits<float>::infinity();
        }

        int start_ix = round(start_x);
        int start_iy = round(start_y);
        if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) {
            return;
        }

        const int MAX_WALLS = 20;
        const int N = width * height;

        // distance dominance table (per cell per wall-count) + best total cost per cell
        vector<vector<float>> min_dist(N, vector<float>(MAX_WALLS, numeric_limits<float>::infinity()));
        vector<float> min_cost(N, numeric_limits<float>::infinity());

        priority_queue<Label, vector<Label>, greater<Label>> pq;

        int start_idx = start_iy * width + start_ix;
        bool start_is_wall = grid[start_idx] > 0;

        // TL of a pixel: pixel_tl map if given, else wall_tl for wall pixels, else 0
        auto pixel_tl_of = [&](int idx) -> float {
            if (pixel_tl != nullptr) {
                return pixel_tl[idx];
            }
            return grid[idx] > 0 ? wall_tl : 0.0f;
        };

        float start_tl = start_is_wall ? pixel_tl_of(start_idx) : 0.0f;
        float init_cost = 20.0f * log10(0.0f + mic_distance) + start_tl;
        min_dist[start_idx][start_is_wall ? 1 : 0] = 0.0f;
        min_cost[start_idx] = init_cost;
        pq.push({0.0f, start_is_wall ? 1 : 0, start_idx, start_tl, init_cost});

        // Convert targets to indices
        vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int tx = round(target_xs[i]);
            int ty = round(target_ys[i]);
            target_indices[i] = (tx >= 0 && tx < width && ty >= 0 && ty < height)
                                    ? (ty * width + tx)
                                    : -1;
        }

        int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
        float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

        while (!pq.empty()) {
            Label curr = pq.top();
            pq.pop();

            if (curr.cost > min_cost[curr.index] + 1e-4f) {
                continue; // stale label
            }

            int cx = curr.index % width;
            int cy = curr.index / width;
            float curr_tl = pixel_tl_of(curr.index);

            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + dx[dir];
                int ny = cy + dy[dir];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                int nidx = ny * width + nx;
                bool next_is_wall = grid[nidx] > 0;
                bool curr_is_wall = grid[curr.index] > 0;

                int nwalls = curr.walls;
                if (next_is_wall && !curr_is_wall) {
                    nwalls += 1;
                }
                if (nwalls >= MAX_WALLS) {
                    continue;
                }

                float ndist = curr.dist + step_dist[dir] * resolution;

                // distance dominance: existing path with <= distance and <= walls wins
                bool dominated = false;
                for (int w = 0; w <= nwalls; ++w) {
                    if (min_dist[nidx][w] <= ndist) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    continue;
                }
                min_dist[nidx][nwalls] = ndist;

                // TL is paid on the air -> material transition (once per barrier),
                // so thick walls cost wall_tl once and open doors (TL 0) cost nothing.
                float next_tl = pixel_tl_of(nidx);
                float transition_tl = (curr_tl <= 0.0f && next_tl > 0.0f) ? next_tl : 0.0f;
                float new_tl = curr.tl + transition_tl;

                float ncost = 20.0f * log10(ndist + mic_distance) + new_tl;
                if (ncost < min_cost[nidx]) {
                    min_cost[nidx] = ncost;
                    pq.push({ndist, nwalls, nidx, new_tl, ncost});
                }
            }
        }

        for (int i = 0; i < num_targets; ++i) {
            int tidx = target_indices[i];
            if (tidx != -1 && min_cost[tidx] != numeric_limits<float>::infinity()) {
                out_attenuations[i] = min_cost[tidx];
            }
        }
    }
}
