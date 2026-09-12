#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

using namespace std;

static inline bool fast_line_of_sight(
    const uint8_t* grid, const float* pixel_tl, int width, int height,
    int x0, int y0, int x1, int y1
) {
    int end_idx = y1 * width + x1;
    if (grid[end_idx] > 0) return false;
    if (pixel_tl != nullptr && pixel_tl[end_idx] > 0.0f) return false;

    int ddx = abs(x1 - x0);
    int ddy = abs(y1 - y0);

    // Adjacent (1 grid step)
    if (ddx <= 1 && ddy <= 1) {
        return true;
    }

    int sx  = (x0 < x1) ? 1 : -1;
    int sy  = (y0 < y1) ? 1 : -1;
    int err = ddx - ddy;
    int x = x0, y = y0;
    while (true) {
        int idx = y * width + x;
        if (grid[idx] > 0) return false;
        if (pixel_tl != nullptr && pixel_tl[idx] > 0.0f) return false;
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 > -ddy) { err -= ddy; x += sx; }
        if (e2 <  ddx) { err += ddx; y += sy; }
    }
    return true;
}

struct Label {
    float dist;        // accumulated Euclidean distance (metres) to THIS node
    int   walls;       // number of air->wall transitions so far
    int   index;       // grid pixel index (y*width + x)
    float tl;          // accumulated transmission loss (dB) along the path
    float cost;        // 20*log10(dist + mic_distance) + tl  (priority key)
    int   parent_x;    // pixel x of THIS label's parent node (for Theta*)
    int   parent_y;    // pixel y of THIS label's parent node (for Theta*)
    float parent_dist; // accumulated distance at the parent node (for Theta*)

    bool operator>(const Label& other) const {
        return cost > other.cost;
    }
};

struct AcousticSolverWorkspace {
    static const int MAX_WALLS = 12;
    int capacity = 0;
    vector<float> min_dist;        // size: capacity * MAX_WALLS
    vector<float> min_cost;        // size: capacity
    vector<uint8_t> target_status; // size: capacity (0: none, 1: unsettled, 2: settled)
    vector<int> touched;

    void ensure_capacity(int N) {
        if (capacity < N) {
            capacity = N;
            min_dist.assign((size_t)N * MAX_WALLS, numeric_limits<float>::infinity());
            min_cost.assign(N, numeric_limits<float>::infinity());
            target_status.assign(N, (uint8_t)0);
            touched.clear();
            touched.reserve(min(N, 65536));
        }
    }

    void reset() {
        if (touched.size() > (size_t)capacity / 2) {
            fill(min_cost.begin(), min_cost.end(), numeric_limits<float>::infinity());
            fill(min_dist.begin(), min_dist.end(), numeric_limits<float>::infinity());
            fill(target_status.begin(), target_status.end(), (uint8_t)0);
        } else {
            for (int idx : touched) {
                min_cost[idx] = numeric_limits<float>::infinity();
                size_t base = (size_t)idx * MAX_WALLS;
                for (int w = 0; w < MAX_WALLS; ++w) {
                    min_dist[base + w] = numeric_limits<float>::infinity();
                }
                target_status[idx] = 0;
            }
        }
        touched.clear();
    }
};

static thread_local AcousticSolverWorkspace ws;

static const int ddx_arr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
static const int ddy_arr[] = { 0, 0, -1, 1, -1,  1, -1, 1};
static const float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f,
                                  1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

template <bool EARLY_EXIT>
static inline void run_acoustic_dijkstra(
    const uint8_t* grid,
    int width,
    int height,
    float resolution,
    float start_x,
    float start_y,
    float wall_tl,
    float mic_distance,
    const float* pixel_tl,
    int num_unsettled
) {
    int start_ix = (int)round(start_x);
    int start_iy = (int)round(start_y);
    if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) {
        return;
    }

    int start_idx = start_iy * width + start_ix;
    bool start_is_wall = grid[start_idx] > 0;
    float start_tl    = start_is_wall ? (pixel_tl ? pixel_tl[start_idx] : wall_tl) : 0.0f;
    float init_cost   = 20.0f * log10f(0.0f + mic_distance) + start_tl;
    int   start_walls = start_is_wall ? 1 : 0;

    size_t start_base = (size_t)start_idx * AcousticSolverWorkspace::MAX_WALLS;
    for (int w = start_walls; w < AcousticSolverWorkspace::MAX_WALLS; ++w) {
        ws.min_dist[start_base + w] = 0.0f;
    }
    ws.min_cost[start_idx] = init_cost;
    ws.touched.push_back(start_idx);

    priority_queue<Label, vector<Label>, greater<Label>> pq;
    pq.push({0.0f, start_walls, start_idx, start_tl, init_cost,
             start_ix, start_iy, 0.0f});

    while (!pq.empty()) {
        Label curr = pq.top();
        pq.pop();

        if (curr.cost > ws.min_cost[curr.index] + 1e-4f) {
            continue;
        }

        if (EARLY_EXIT) {
            if (ws.target_status[curr.index] == 1) {
                ws.target_status[curr.index] = 2; // settled
                num_unsettled--;
                if (num_unsettled <= 0) {
                    break;
                }
            }
        }

        int cx = curr.index % width;
        int cy = curr.index / width;

        int   par_x    = curr.parent_x;
        int   par_y    = curr.parent_y;
        float par_dist = curr.parent_dist;
        bool curr_is_wall = grid[curr.index] > 0;

        for (int dir = 0; dir < 8; ++dir) {
            int nx = cx + ddx_arr[dir];
            int ny = cy + ddy_arr[dir];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            int nidx = ny * width + nx;
            bool next_is_wall = grid[nidx] > 0;
            int nwalls = curr.walls + ((next_is_wall && !curr_is_wall) ? 1 : 0);
            if (nwalls >= AcousticSolverWorkspace::MAX_WALLS) {
                continue;
            }

            float next_tl_val = pixel_tl ? pixel_tl[nidx] : (next_is_wall ? wall_tl : 0.0f);
            bool can_los = (!next_is_wall && next_tl_val <= 0.0f);

            float ndist;
            int   new_par_x, new_par_y;
            float new_par_dist;
            float next_tl_contrib;

            if (can_los && fast_line_of_sight(grid, pixel_tl, width, height, par_x, par_y, nx, ny)) {
                float fdx = (float)(nx - par_x);
                float fdy = (float)(ny - par_y);
                ndist         = par_dist + resolution * sqrtf(fdx * fdx + fdy * fdy);
                new_par_x     = par_x;
                new_par_y     = par_y;
                new_par_dist  = par_dist;
                next_tl_contrib = 0.0f;
            } else {
                ndist         = curr.dist + step_dist[dir] * resolution;
                new_par_x     = cx;
                new_par_y     = cy;
                new_par_dist  = curr.dist;
                float curr_tl_val = pixel_tl ? pixel_tl[curr.index] : (curr_is_wall ? wall_tl : 0.0f);
                next_tl_contrib   = (curr_tl_val <= 0.0f && next_tl_val > 0.0f) ? next_tl_val : 0.0f;
            }

            size_t n_base = (size_t)nidx * AcousticSolverWorkspace::MAX_WALLS;
            if (ws.min_dist[n_base + nwalls] <= ndist) {
                continue;
            }
            for (int w = nwalls; w < AcousticSolverWorkspace::MAX_WALLS; ++w) {
                if (ws.min_dist[n_base + w] > ndist) {
                    ws.min_dist[n_base + w] = ndist;
                } else {
                    break;
                }
            }

            float new_tl = curr.tl + next_tl_contrib;
            float ncost  = 20.0f * log10f(ndist + mic_distance) + new_tl;
            if (ncost < ws.min_cost[nidx]) {
                if (ws.min_cost[nidx] == numeric_limits<float>::infinity()) {
                    ws.touched.push_back(nidx);
                }
                ws.min_cost[nidx] = ncost;
                pq.push({ndist, nwalls, nidx, new_tl, ncost,
                         new_par_x, new_par_y, new_par_dist});
            }
        }
    }
}

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
        const float* pixel_tl,
        float* out_attenuations
    ) {
        for (int i = 0; i < num_targets; ++i) {
            out_attenuations[i] = numeric_limits<float>::infinity();
        }

        const int N = width * height;
        ws.ensure_capacity(N);
        ws.reset();

        int num_unsettled = 0;
        vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int tx = (int)round(target_xs[i]);
            int ty = (int)round(target_ys[i]);
            if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
                int tidx = ty * width + tx;
                target_indices[i] = tidx;
                if (ws.target_status[tidx] == 0) {
                    ws.target_status[tidx] = 1;
                    num_unsettled++;
                }
            } else {
                target_indices[i] = -1;
            }
        }

        if (num_unsettled > 0) {
            run_acoustic_dijkstra<true>(
                grid, width, height, resolution,
                start_x, start_y, wall_tl, mic_distance, pixel_tl,
                num_unsettled
            );
        }

        for (int i = 0; i < num_targets; ++i) {
            int tidx = target_indices[i];
            if (tidx != -1) {
                if (ws.min_cost[tidx] != numeric_limits<float>::infinity()) {
                    out_attenuations[i] = ws.min_cost[tidx];
                }
                ws.target_status[tidx] = 0;
            }
        }

        ws.reset();
    }

    void solve_acoustic_grid(
        const uint8_t* grid,
        int width,
        int height,
        float resolution,
        float start_x,
        float start_y,
        float wall_tl,
        float mic_distance,
        const float* pixel_tl,
        float* out_field
    ) {
        const int N = width * height;
        ws.ensure_capacity(N);
        ws.reset();

        run_acoustic_dijkstra<false>(
            grid, width, height, resolution,
            start_x, start_y, wall_tl, mic_distance, pixel_tl,
            0
        );

        for (int i = 0; i < N; ++i) {
            out_field[i] = ws.min_cost[i];
        }

        ws.reset();
    }
}
