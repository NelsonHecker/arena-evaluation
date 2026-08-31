import os
import yaml
import pathlib
import subprocess
from PIL import Image

class MapRegistry:
    @staticmethod
    def _find_ros_map_dir(map_name: str, run_dir: pathlib.Path | None = None) -> pathlib.Path | None:
        candidates = [map_name]
        if "_stage" in map_name:
            candidates.append(map_name.split("_stage")[0])
        if "-" in map_name:
            candidates.append(map_name.split("-")[0])
        if "_" in map_name:
            candidates.append(map_name.split("_")[0])

        for m_name in candidates:
            if run_dir is not None:
                for cand_p in [
                    pathlib.Path(run_dir) / "worlds" / m_name,
                    pathlib.Path(run_dir) / "maps" / m_name,
                ]:
                    if cand_p.exists():
                        return cand_p

            try:
                res = subprocess.run(["rospack", "find", "arena_simulation_setup"],
                                     capture_output=True, text=True, check=False)
                if res.returncode == 0 and res.stdout.strip():
                    map_dir = pathlib.Path(res.stdout.strip()) / "worlds" / m_name
                    if map_dir.exists():
                        return map_dir
            except Exception:
                pass

            try:
                from ament_index_python.packages import get_package_share_directory

                pkg_path = pathlib.Path(get_package_share_directory("arena_simulation_setup"))
                map_dir = pkg_path / "worlds" / m_name
                if map_dir.exists():
                    return map_dir
            except Exception:
                pass

            for ws in ("/opt/arena_ws", "/home/nelson/arena_ws"):
                map_dir = pathlib.Path(ws) / "src" / "Arena" / "arena_simulation_setup" / "worlds" / m_name
                if map_dir.exists():
                    return map_dir

            arena_dir = os.environ.get("ARENA_DIR")
            if arena_dir:
                map_dir = pathlib.Path(arena_dir) / "arena_simulation_setup/worlds" / m_name
                if map_dir.exists():
                    return map_dir

        return None

    @staticmethod
    def get_map(map_name: str, cache_dir: pathlib.Path | None = None, run_dir: pathlib.Path | None = None) -> dict | None:
        if not map_name or map_name == "unknown":
            return None
            
        if cache_dir is None:
            data_root = os.environ.get("ARENA_DATA_DIR") or os.path.join(os.environ.get("ARENA_WS_DIR", "/opt/arena_ws"), "data")
            cache_dir = pathlib.Path(data_root) / "maps_cache"
            
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        png_path = cache_dir / f"{map_name}.png"
        meta_path = cache_dir / f"{map_name}.yaml"
        
        if png_path.exists() and meta_path.exists():
            with open(meta_path, "r") as f:
                return yaml.safe_load(f)
                
        if run_dir is not None:
            try:
                ros_map_dir = MapRegistry._find_ros_map_dir(map_name, run_dir=run_dir)
            except TypeError:
                ros_map_dir = MapRegistry._find_ros_map_dir(map_name)
        else:
            ros_map_dir = MapRegistry._find_ros_map_dir(map_name)
        if not ros_map_dir:
            return None
            
        yaml_path = None
        for cand in ["map/map.yaml", "0/map.yaml", "map.yaml"]:
            if (ros_map_dir / cand).exists():
                yaml_path = ros_map_dir / cand
                break
                
        if not yaml_path:
            return None
            
        with open(yaml_path, "r") as f:
            map_meta = yaml.safe_load(f)
            
        image_name = map_meta.get("image", "map.pgm")
        image_path = yaml_path.parent / image_name
        
        if not image_path.exists():
            return None
            
        try:
            img = Image.open(image_path)
            img = img.convert("RGBA")
            img.save(png_path)
            
            origin = [float(x) for x in map_meta.get("origin", [0.0, 0.0, 0.0])]
            
            cache_meta = {
                "png_path": str(png_path),
                "resolution": float(map_meta.get("resolution", 0.05)),
                "origin": origin,
                "width": img.width,
                "height": img.height
            }
            with open(meta_path, "w") as f:
                yaml.dump(cache_meta, f)
                
            return cache_meta
        except Exception as e:
            print(f"Failed to cache map {map_name}: {e}")
            return None
