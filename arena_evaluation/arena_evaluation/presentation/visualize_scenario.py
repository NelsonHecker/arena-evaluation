#!/usr/bin/env python3
"""Publication-Grade 2D Scenario Visualization Tool for Arena Evaluation 3.0.

Renders occupancy grid maps overlaid with:
- Structural walls and partition doors (with TL transmission loss badges)
- Physical 3D furniture entities (bookcase stacks, hospital beds, desks, cabinets)
- Semantic speed and acoustic quiet zones
- Robot missions (start quivers, via-waypoints, final goals, planned paths)
- Dynamic pedestrian patrols and flow trajectories
- Stationary acoustic microphone arrays (virtual sensor probes)
Generates high-resolution (220 DPI) publication-ready figures matching ICRA 2027 standards.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml


def visualize_scenario(
    map_yaml_path: str | Path,
    scenario_yaml_path: str | Path,
    output_png_path: str | Path,
    world_yaml_path: str | Path | None = None,
    scenario_title: str | None = None,
) -> None:
    map_yaml_path = Path(map_yaml_path)
    scenario_yaml_path = Path(scenario_yaml_path)
    output_png_path = Path(output_png_path)

    # Automatically resolve world.yaml if not provided
    if world_yaml_path is None:
        candidate = map_yaml_path.parent / "world.yaml"
        if candidate.exists():
            world_yaml_path = candidate
    else:
        world_yaml_path = Path(world_yaml_path)

    # 1. Load Map Metadata
    with open(map_yaml_path, "r", encoding="utf-8") as f:
        map_meta = yaml.safe_load(f)

    res = float(map_meta["resolution"])
    origin = map_meta["origin"]  # [ox, oy, yaw]
    img_rel = map_meta["image"]
    img_path = map_yaml_path.parent / img_rel

    # 2. Load Occupancy Image
    occ_img = Image.open(img_path).convert("L")
    occ_arr = np.array(occ_img)
    H, W = occ_arr.shape

    # Normalize image: free space -> white (255), unknown/outside -> black (0)
    # If image has unknown (205), turn it black for crisp floorplan contrast
    disp_arr = np.zeros_like(occ_arr)
    disp_arr[occ_arr > 240] = 255  # Free interior space is white
    disp_arr[occ_arr < 50] = 0    # Walls are black

    # Coordinate transform helper (World meters to Pixel coordinates)
    def w2p(x: float, y: float) -> tuple[float, float]:
        px = (x - origin[0]) / res
        py = (H - 1) - (y - origin[1]) / res
        return px, py

    # 3. Load Scenario YAML
    with open(scenario_yaml_path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    # 4. Load World YAML (for 3D physical entities & doors)
    world_data = None
    if world_yaml_path and world_yaml_path.exists():
        with open(world_yaml_path, "r", encoding="utf-8") as f:
            world_data = yaml.safe_load(f)

    fig_w = 14
    fig_h = max(7, int(14 * H / W))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.imshow(disp_arr, cmap="gray", origin="upper")

    # 5. Render Semantic Regions (Speed Limits & Quiet Zones)
    region_styles = {
        "quiet": {"color": "#D4E6F1", "alpha": 0.45},
        "study": {"color": "#E8F8F5", "alpha": 0.45},
        "circulation": {"color": "#FEF9E7", "alpha": 0.50},
        "reception": {"color": "#EAF2F8", "alpha": 0.50},
        "transit": {"color": "#E8F6F3", "alpha": 0.50},
        "corridor": {"color": "#EAECEE", "alpha": 0.40},
    }
    for reg_name, reg_data in scenario.get("regions", {}).items():
        corners = reg_data.get("corners", [])
        if corners:
            px_corners = [w2p(c[0], c[1]) for c in corners]
            max_v = reg_data.get("max_velocity")
            # Determine color style
            style = {"color": "#D6EAF8", "alpha": 0.40}
            for key, val in region_styles.items():
                if key in reg_name.lower():
                    style = val
                    break
            lbl = f"Zone: {reg_name} (max {max_v} m/s)" if max_v else f"Zone: {reg_name}"
            poly = plt.Polygon(px_corners, color=style["color"], alpha=style["alpha"], label=lbl, zorder=2)
            ax.add_patch(poly)

    # 6. Render Physical Static Entities from world.yaml (Shelves, Beds, Desks)
    has_bookcase_legend = False
    has_bed_legend = False
    has_desk_legend = False
    has_door_legend = False

    if world_data and "zones" in world_data:
        for zone in world_data["zones"]:
            # Render Doors
            for door in zone.get("doors", []):
                d_start = door.get("start", {})
                d_end = door.get("end", {})
                if d_start and d_end:
                    spx, spy = w2p(d_start["x"], d_start["y"])
                    epx, epy = w2p(d_end["x"], d_end["y"])
                    mpx = (spx + epx) / 2.0
                    mpy = (spy + epy) / 2.0
                    ax.plot([spx, epx], [spy, epy], color="#E74C3C", linewidth=3.5, zorder=5)
                    lbl = "Partition Door (TL=28dB)" if not has_door_legend else ""
                    ax.plot(mpx, mpy, "D", color="#FF7675", markersize=8, markeredgecolor="black", label=lbl, zorder=6)
                    has_door_legend = True

            # Render Static Furniture Entities
            entities = zone.get("entities", {}).get("static", [])
            for ent in entities:
                model = ent.get("model", "")
                pos = ent.get("pose", {}).get("position", {})
                if not pos:
                    continue
                ex, ey = pos["x"], pos["y"]
                epx, epy = w2p(ex, ey)

                # Bookcases (Office/SM_BookcaseA)
                if "Bookcase" in model:
                    # Realistic bookcase footprint: 1.0m width, 0.4m depth
                    bw_px = 1.0 / res
                    bh_px = 7.5 / res  # Multi-unit bookcase row block
                    rect = patches.Rectangle(
                        (epx - bw_px / 2.0, epy - bh_px / 2.0),
                        bw_px, bh_px,
                        linewidth=1.2, edgecolor="#4A235A", facecolor="#8B4513", alpha=0.85, zorder=4
                    )
                    ax.add_patch(rect)
                    if not has_bookcase_legend:
                        ax.plot([], [], "s", color="#8B4513", markeredgecolor="#4A235A", label="Library Bookcases (SM_BookcaseA)")
                        has_bookcase_legend = True

                # Hospital Beds (Hospital/SM_HospitalBed_01b)
                elif "Bed" in model and "Side" not in model:
                    bw_px = 1.2 / res
                    bh_px = 2.2 / res
                    rect = patches.Rectangle(
                        (epx - bw_px / 2.0, epy - bh_px / 2.0),
                        bw_px, bh_px,
                        linewidth=1.2, edgecolor="#1B4F72", facecolor="#85C1E9", alpha=0.90, zorder=4
                    )
                    ax.add_patch(rect)
                    # Pillow indicator
                    pillow = patches.Rectangle(
                        (epx - bw_px * 0.35, epy - bh_px * 0.45),
                        bw_px * 0.7, bh_px * 0.25,
                        facecolor="white", edgecolor="#1B4F72", zorder=5
                    )
                    ax.add_patch(pillow)
                    if not has_bed_legend:
                        ax.plot([], [], "s", color="#85C1E9", markeredgecolor="#1B4F72", label="Hospital Bed (SM_HospitalBed)")
                        has_bed_legend = True

                # Desks / Work Tables (Stand_Table_Work, SM_TableWorkingDouble, Reception)
                elif "Table" in model or "Desk" in model or "Reception" in model:
                    dw_px = 1.6 / res
                    dh_px = 0.9 / res
                    rect = patches.Rectangle(
                        (epx - dw_px / 2.0, epy - dh_px / 2.0),
                        dw_px, dh_px,
                        linewidth=1.0, edgecolor="#1C2833", facecolor="#BDC3C7", alpha=0.85, zorder=4
                    )
                    ax.add_patch(rect)
                    if not has_desk_legend:
                        ax.plot([], [], "s", color="#BDC3C7", markeredgecolor="#1C2833", label="Work Desk / Table")
                        has_desk_legend = True

    # 7. Render Robot Start, Waypoints, Goals, and Path
    for r_idx, r in enumerate(scenario.get("robots", [])):
        sx, sy, syaw = r["start"]
        spx, spy = w2p(sx, sy)
        ax.plot(spx, spy, "o", color="#27AE60", markersize=10, markeredgecolor="black", label="Robot Start" if r_idx == 0 else "", zorder=8)
        ax.quiver(spx, spy, np.cos(syaw), -np.sin(syaw), color="#1E8449", scale=18, width=0.007, zorder=8)

        # Plot full path through phases
        path_pts = [(sx, sy)]
        for phase in r.get("phases", []):
            if "goto" in phase:
                path_pts.append((phase["goto"][0], phase["goto"][1]))

        if len(path_pts) > 1:
            px_pts = [w2p(pt[0], pt[1]) for pt in path_pts]
            xs, ys = zip(*px_pts)
            ax.plot(xs, ys, "--", color="#27AE60", linewidth=2.2, alpha=0.85, label="Planned Trajectory" if r_idx == 0 else "", zorder=7)

        # Plot Via-Waypoints and Final Goal
        phases = r.get("phases", [])
        for p_idx, phase in enumerate(phases):
            if "goto" in phase:
                gx, gy, _ = phase["goto"]
                gpx, gpy = w2p(gx, gy)
                is_final = (p_idx == len(phases) - 1)
                if is_final:
                    ax.plot(gpx, gpy, "*", color="#C0392B", markersize=14, markeredgecolor="black", label="Mission Goal" if r_idx == 0 else "", zorder=9)
                elif len(phases) <= 5:
                    ax.plot(gpx, gpy, "^", color="#F1C40F", markersize=9, markeredgecolor="black", label="Via-Waypoint" if (p_idx == 0 and r_idx == 0) else "", zorder=8)
                else:
                    # Continuous multi-pick: plot pick targets with distinct small star
                    ax.plot(gpx, gpy, "*", color="#E67E22", markersize=10, markeredgecolor="black", zorder=8)

    # 8. Render Dynamic Pedestrians & Patrol Routes
    ped_colors = ["#E67E22", "#8E44AD", "#16A085", "#D35400"]
    for p_idx, ped in enumerate(scenario.get("dynamic", [])):
        c = ped_colors[p_idx % len(ped_colors)]
        px, py, pyaw = ped["pose"]
        ppx, ppy = w2p(px, py)
        ped_name = ped.get("name", f"ped_{p_idx}")
        ax.plot(ppx, ppy, "o", color=c, markersize=8, markeredgecolor="black", label=f"Ped: {ped_name}", zorder=8)
        ax.quiver(ppx, ppy, np.cos(pyaw), -np.sin(pyaw), color=c, scale=18, width=0.006, zorder=8)

        wps = ped.get("waypoints", [])
        if wps:
            all_pts = [(px, py)] + [tuple(w[:2]) for w in wps]
            pix_pts = [w2p(pt[0], pt[1]) for pt in all_pts]
            xs, ys = zip(*pix_pts)
            ax.plot(xs, ys, ":", color=c, linewidth=2.0, alpha=0.90, zorder=7)

    # 9. Render Static Microphone Probes (Receptors)
    for m_idx, static_obj in enumerate(scenario.get("static", [])):
        name = static_obj.get("name", f"mic_{m_idx}")
        pose = static_obj.get("pose", [0.0, 0.0, 0.0])
        mx, my = pose[0], pose[1]
        mpx, mpy = w2p(mx, my)
        ax.plot(mpx, mpy, "s", color="#F39C12", markersize=9, markeredgecolor="black",
                label="Microphone Probe / Noise Sink" if m_idx == 0 else "", zorder=9)
        ax.text(mpx + 6, mpy - 7, name, color="#B9770E", fontsize=8.0, fontweight="bold", zorder=10)

    # 10. Title & Legend
    title = scenario_title or f"Scenario: {scenario_yaml_path.parent.name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    ax.legend(loc="upper right", bbox_to_anchor=(1.26, 1.0), framealpha=0.95, fontsize=8.0, ncol=1)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Generated publication visualization: {output_png_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="Path to map.yaml")
    ap.add_argument("--scenario", required=True, help="Path to scenario.yaml")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--world", default=None, help="Optional path to world.yaml")
    ap.add_argument("--title", default=None, help="Scenario title")
    args = ap.parse_args()

    visualize_scenario(args.map, args.scenario, args.out, args.world, args.title)


if __name__ == "__main__":
    main()
