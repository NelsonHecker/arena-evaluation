from __future__ import annotations

import ast
import asyncio
import dataclasses
import enum
import functools
import importlib
import pathlib
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import pydantic
from arena_mcp.core import ToolError

from .base import YamlText, config_stem, domain, parse_yaml, run_path
from .catalog import config_identifier, config_names

if TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.task_layer import TaskLayer, TaskModes
    from arena_simulation_setup.tree import DomainAssetIdentifier
    from rcl_interfaces.msg import ParameterDescriptor

    from arena_evaluation.benchmark.config import Contest

Overwrite = Annotated[bool, pydantic.Field(description="replace an existing config of that name")]


class _Declarations:
    """Collects what a task-mode schema declares, standing in for the node it normally declares on."""

    def __init__(self) -> None:
        self.rosparam = self
        self.declared: list[tuple[str, ParameterDescriptor]] = []

    def declare_forward(self, name: str, value: object, *, descriptor: ParameterDescriptor) -> None:
        self.declared.append((name, descriptor))


def _constraint_tokens(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in filter(None, text.split(";")):
        key, sep, value = token.partition(":")
        out[key] = (value.split(",") if key == "enum" else value) if sep else True
    return out


def mode_params(layer: TaskLayer, selected: TaskModes) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """The parameters the selected modes of the task layer declare below their namespace with their constraint tokens, by axis and mode."""
    namespaces = layer.mode_namespaces()

    def declared(axis: str, mode: str) -> list[dict[str, Any]]:
        collector = _Declarations()
        layer.declare_mode_schema(cast("ROSParamServer", collector), axis, mode)
        prefix = f"{namespaces[axis][mode]}."
        return [{"name": name.removeprefix(prefix), **_constraint_tokens(descriptor.additional_constraints)} for name, descriptor in collector.declared]

    return {axis: {mode: declared(axis, mode) for mode in modes} for axis, modes in selected.items()}


def domain_asset_names(identifier: type[DomainAssetIdentifier]) -> list[str]:
    """Names of a domain asset kind the way task modes reference them."""
    from arena_simulation_setup import DOMAIN_DEFAULT

    return sorted({item.name if item.domain == DOMAIN_DEFAULT else item.shortname for item in identifier.listall()})


def world_names() -> list[str]:
    from arena_simulation_setup.tree.World import WorldIdentifier

    return sorted({item.shortname for item in WorldIdentifier.listall() if not item.shortname.startswith(".")})


def world_resolves(name: str, known: set[str]) -> bool:
    """True when every world of a possibly composite, level-filtered world name is known."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import composite_parts

    return all(WorldIdentifier.split_filter(part)[0] in known for part in composite_parts(name))


def registered_drivers() -> dict[str, set[str]]:
    """Drivers per cap from the bringup and adapter registries."""
    from arena_robots.bringup import BRINGUPS
    from arena_runtime.env.adapters import ADAPTERS

    for cap in ADAPTERS:
        importlib.import_module(f"arena_runtime.env.adapters.{cap}")
    return {cap: set(BRINGUPS.get(cap, {}).keys()) | set(ADAPTERS.get(cap, {}).keys()) for cap in sorted(set(BRINGUPS) | set(ADAPTERS))}


def _leaves(block: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and not ({"min", "max"} <= set(value)):
            out += _leaves(value, f"{name}.")
        elif isinstance(value, dict):
            out.append(f"{name}.n")
            out += _leaves({key: item for key, item in value.items() if key not in ("min", "max")}, f"{name}.")
        else:
            out.append(name)
    return out


def _at(block: dict[str, Any], dotted: str) -> object:
    node: object = block
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LaunchSpec:
    """The declared type of one launch argument, and its allowed values when the declaration is an enum."""

    expected: type
    choices: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LaunchRegistry:
    """Every declared launch argument and the typed subset the authoring validator can check."""

    declared: frozenset[str]
    specs: dict[str, LaunchSpec]


def _python_type(type_: object, parse: object, parameter_type: object, default: object) -> LaunchSpec:
    """Resolve a declared launch argument to the python type the launcher parses."""
    if isinstance(parse, type) and issubclass(parse, enum.Enum):
        return LaunchSpec(expected=str, choices=tuple(str(member.value) for member in parse))
    if isinstance(type_, enum.Enum):
        mapped = {"STRING": str, "BOOL": bool, "INTEGER": int, "DOUBLE": float}.get(type_.name)
        if mapped is not None:
            return LaunchSpec(expected=mapped)
    if parameter_type in (bool, int, float, str):
        return LaunchSpec(expected=parameter_type)
    if isinstance(parameter_type, type) and issubclass(parameter_type, enum.Enum):
        return LaunchSpec(expected=str, choices=tuple(str(member.value) for member in parameter_type))
    if isinstance(default, bool):
        return LaunchSpec(expected=bool)
    if isinstance(default, int):
        return LaunchSpec(expected=int)
    if isinstance(default, float):
        return LaunchSpec(expected=float)
    if isinstance(default, str):
        return LaunchSpec(expected=str)
    return LaunchSpec(expected=object)


def _configuration_launch_specs() -> tuple[dict[str, LaunchSpec], frozenset[str]]:
    """The parameters the env configuration declares, captured with their types instead of declared on a node."""
    from arena_runtime.env.constants.runtime import Configuration

    specs: dict[str, LaunchSpec] = {}

    class _Recorder:
        _parameter_type: object = object

        def __class_getitem__(cls, item: object) -> type:
            return type("_RecordedParam", (cls,), {"_parameter_type": item})

        def __init__(self, name: str, value: object = None, *, type_: object = None, parse: object = None, **_kwargs: object) -> None:
            specs[str(name)] = _python_type(type_, parse, self._parameter_type, value)

    class _Server:
        ROSParam = _Recorder

    Configuration(_Server())
    return specs, frozenset(specs)


def _infer_launch_default(node: ast.expr | None) -> LaunchSpec | None:
    if not isinstance(node, ast.Constant):
        return None
    value = node.value
    if isinstance(value, bool):
        return LaunchSpec(expected=bool)
    if isinstance(value, int):
        return LaunchSpec(expected=int)
    if isinstance(value, float):
        return LaunchSpec(expected=float)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return LaunchSpec(expected=bool)
        try:
            int(value)
        except ValueError:
            try:
                float(value)
            except ValueError:
                return None
            return LaunchSpec(expected=float)
        return LaunchSpec(expected=int)
    return None


def _launch_file_specs() -> tuple[dict[str, LaunchSpec], frozenset[str]]:
    """Launch arguments declared by the bringup launch files, typed where the declared default shows the type."""
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    specs: dict[str, LaunchSpec] = {}
    declared: set[str] = set()
    for package in ("arena_bringup", "arena_runtime", "arena_simulation_setup"):
        try:
            base = pathlib.Path(get_package_share_directory(package)) / "launch"
        except PackageNotFoundError:
            continue
        for file in sorted(base.rglob("*.launch.py")):
            try:
                tree = ast.parse(file.read_text())
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
                if name not in {"LaunchArgument", "DeclareLaunchArgument"}:
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                name_node = keywords.get("name")
                if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
                    continue
                argument = name_node.value
                declared.add(argument)
                spec = _infer_launch_default(keywords.get("default_value"))
                if spec is not None and argument not in specs:
                    specs[argument] = spec
    return specs, frozenset(declared)


@functools.lru_cache(maxsize=1)
def launch_registry() -> LaunchRegistry:
    """Declared launch arguments with the types the runtime configuration and bringup launch files fix."""
    specs: dict[str, LaunchSpec] = {}
    declared: set[str] = set()
    file_specs, file_declared = _launch_file_specs()
    specs.update(file_specs)
    declared.update(file_declared)
    config_specs, config_declared = _configuration_launch_specs()
    specs.update(config_specs)
    declared.update(config_declared)
    return LaunchRegistry(declared=frozenset(declared), specs=specs)


_BOOL_WORDS = frozenset({"true", "false", "1", "0", "yes", "no", "on", "off"})


def _launch_type_problem(where: str, name: str, value: object, spec: LaunchSpec) -> str | None:
    """Why a launch value does not match its declared type, typed booleans never accepting the string spellings."""
    expected = spec.expected
    if expected is bool:
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and value.strip().lower() in _BOOL_WORDS:
            return f"{where}: launch arg {name!r} must be a boolean, got the string {value!r}; write YAML true/false unquoted"
        return f"{where}: launch arg {name!r} must be a boolean, got {type(value).__name__} {value!r}"
    if expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return f"{where}: launch arg {name!r} must be an integer, got {type(value).__name__} {value!r}" + (" (remove the quotes)" if isinstance(value, str) else "")
    if expected is float:
        if isinstance(value, float | int) and not isinstance(value, bool):
            return None
        return f"{where}: launch arg {name!r} must be a number, got {type(value).__name__} {value!r}" + (" (remove the quotes)" if isinstance(value, str) else "")
    if expected is str:
        if isinstance(value, str):
            return None
        return f"{where}: launch arg {name!r} must be a string, got {type(value).__name__} {value!r}"
    if spec.choices and isinstance(value, str) and value not in spec.choices:
        return f"{where}: launch arg {name!r} must be one of {list(spec.choices)}, got {value!r}"
    return None


def launch_arg_problems(where: str, launch_args: dict[str, object]) -> tuple[list[str], list[str]]:
    """Problems and warnings for one suite's launch block against the declared launch arguments and their types."""
    registry = launch_registry()
    problems: list[str] = []
    warnings: list[str] = []
    for name, value in launch_args.items():
        if name not in registry.declared:
            warnings.append(f"{where}: launch arg {name!r} is not declared by the runtime configuration or the bringup launch files")
            continue
        spec = registry.specs.get(name)
        if spec is not None:
            problem = _launch_type_problem(where, name, value, spec)
            if problem is not None:
                problems.append(problem)
    return problems, warnings


def check_suite(data: object) -> dict[str, Any]:
    """Suite.parse verdict plus warnings for task modes, config keys, worlds, robots and models the registries and resolvers do not know."""
    from arena_robots.Robot import RobotIdentifier
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.robot_manager.robots_manager import split_robot_arg
    from arena_runtime.env.task_layer import registered_modes, task_layer
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    from arena_evaluation.benchmark.config import Suite, stage_modes

    if not isinstance(data, dict):
        return {"valid": False, "problems": [f"a suite is a mapping with stages, got {type(data).__name__}"]}
    try:
        suite = Suite.parse("_validate", data)
    except (LookupError, ValueError, TypeError) as exc:
        return {"valid": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    layer = task_layer(suite.task_layer)
    vocabulary = registered_modes(layer)
    chosen = [stage_modes(stage, vocabulary) for stage in suite.stages]
    selected: TaskModes = {axis: sorted({mode for modes in chosen for mode in modes[axis] if mode in vocabulary.get(axis, [])}) for axis in vocabulary}
    declared_params = mode_params(layer, selected)
    worlds = set(world_names())
    robots = {item.shortname for item in RobotIdentifier.listall()}
    catalogs = {"pedestrians": set(domain_asset_names(HumanIdentifier)), "objects": set(domain_asset_names(ObjectIdentifier))}
    human_blocks = {*Constants.HUMAN_PARAM_NAMESPACES.values(), Constants.HUMAN_PARAM_RESERVED}
    problems: list[str] = []
    warnings: list[str] = []
    for raw, stage, stage_chosen in zip(data["stages"], suite.stages, chosen, strict=True):
        where = f"stage {stage.name!r}"
        for axis, names in stage_chosen.items():
            problems.extend(f"{where}: {axis} mode {mode!r} is not registered, see episode_task_modes" for mode in names if mode not in vocabulary.get(axis, []))
        if Constants.HUMAN_PARAM_RESERVED in stage.config:
            problems.append(f"{where}: config block {Constants.HUMAN_PARAM_RESERVED!r} is reserved and fails the stage")
        axis_of = {mode: axis for axis, names in stage_chosen.items() for mode in names}
        for block_name, block in stage.config.items():
            if block_name in human_blocks:
                continue
            if block_name not in axis_of:
                warnings.append(f"{where}: config block {block_name!r} matches none of the stage's modes and is not sent")
                continue
            axis = axis_of[block_name]
            if block_name not in declared_params.get(axis, {}) or not isinstance(block, dict):
                continue
            params = declared_params[axis][block_name]
            declared = {param["name"] for param in params}
            for leaf in _leaves(block):
                if leaf not in declared:
                    warnings.append(f"{where}: config.{block_name}.{leaf} is not a parameter of {block_name}, known: {sorted(declared)}")
            for param in params:
                catalog = catalogs.get(str(param.get("catalog")))
                names = _at(block, param["name"]) if catalog is not None else None
                if catalog is None or not isinstance(names, list):
                    continue
                for name in names:
                    if name not in catalog:
                        warnings.append(f"{where}: config.{block_name}.{param['name']} model {name!r} is not in the {param['catalog']} catalog, fetch it with arena asset pull or fix the name")
        if not world_resolves(stage.map, worlds):
            warnings.append(f"{where}: map {stage.map!r} does not resolve locally, it may be bucket-only (assets_ls kind=world network=true) or misspelled")
        unknown = [entry for entry in split_robot_arg(stage.robot) if entry.partition("[")[0].strip() not in robots]
        if raw.get("robot") != "auto" and unknown:
            warnings.append(f"{where}: robot {', '.join(unknown)} not in the robot catalog (robots_catalog)")
    launch_problems, launch_warnings = launch_arg_problems("suite launch", suite.launch_args)
    problems.extend(launch_problems)
    warnings.extend(launch_warnings)
    return {"valid": not problems, "n_stages": len(suite.stages), "stages": [stage.name for stage in suite.stages], "references": suite.references, "launch": suite.launch_args, "problems": problems, "warnings": warnings}


def _contest_drivers(args: dict[str, Any], cap: str) -> str | None:
    value = args.get(cap)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("driver"), str):
        return value["driver"]
    flat = args.get(f"{cap}.driver")
    return flat if isinstance(flat, str) else None


def check_contest(data: object) -> dict[str, Any]:
    """Contest.parse verdict plus the runner's nav2 preflight and drivers checked against the bringup and adapter registries."""
    from arena_evaluation.benchmark.config import Contest
    from arena_evaluation.benchmark.runner import _preflight_contest

    if not isinstance(data, list | dict):
        return {"valid": False, "problems": [f"a contest is a list or a mapping, got {type(data).__name__}"]}
    try:
        contest = Contest.parse("_validate", data)
    except (KeyError, ValueError, TypeError) as exc:
        return {"valid": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    drivers = registered_drivers()
    problems = list(_preflight_contest(contest))
    problems.extend(_nav2_plugin_problems(contest))
    problems.extend(_planner_availability_problems(contest))
    for contestant in contest.contestants:
        for cap, known in drivers.items():
            driver = _contest_drivers(contestant.args, cap)
            if driver is not None and driver not in known:
                problems.append(f"{contestant.name}: {cap} driver {driver!r} is not registered, known: {sorted(known)}")
    return {"valid": not problems, "n_contestants": len(contest.contestants), "contestants": [{"name": entry.name, "args": entry.args, "parts": entry.parts} for entry in contest.contestants], "problems": problems}


_DRL_DRIVERS: Final = frozenset({"drl", "vla", "rosnav_rl"})


def _cap_planner(args: dict[str, Any], cap: str) -> str | None:
    """The planner a cap block names, e.g. ``mobile: {driver: drl, planner: cadrl}``."""
    value = args.get(cap)
    if isinstance(value, dict):
        planner = value.get("planner")
        return planner if isinstance(planner, str) else None
    return None


def _plugin_yaml(name: str, kind: str) -> dict[str, Any] | None:
    """One nav2 controller or inter-planner config from the arena_robots share dir, None when it is absent."""
    import pathlib

    import yaml
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        path = pathlib.Path(get_package_share_directory("arena_robots")) / "config" / "nav2" / kind / name / f"{kind.removesuffix('s')}_config.yaml"
    except PackageNotFoundError:
        return None
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else None


def _plugins(node: object) -> list[str]:
    """Every ``plugin`` string in a nested nav2 config mapping."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "plugin" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_plugins(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_plugins(value))
    return found


def _nav2_plugin_problems(contest: Contest) -> list[str]:
    """nav2 contestants whose controller/inter-planner config needs a plugin this installation does not ship."""
    from arena_planners import resolver

    try:
        installed = set(resolver.list_available()["nav2"])
    except (OSError, RuntimeError):
        return []
    short = {name.rsplit("::", 1)[-1].lower() for name in installed}
    problems: list[str] = []
    for contestant in contest.contestants:
        mobile = contestant.args.get("mobile")
        if not isinstance(mobile, dict) or mobile.get("driver") != "nav2":
            continue
        for name, kind in ((mobile.get("local_planner"), "controllers"), (mobile.get("inter_planner"), "interplanners")):
            if not isinstance(name, str) or name == "none":
                continue
            config = _plugin_yaml(name, kind)
            if config is None:
                continue
            for plugin in _plugins(config):
                if plugin.rsplit("::", 1)[-1].lower() not in short:
                    problems.append(f"{contestant.name}: {kind[:-1]} {name!r} needs plugin {plugin!r}, which this installation does not ship (installed: {sorted(installed)}); install the nav2 stack or pick an installed {kind[:-1]}")
    return problems


def _training_available() -> bool:
    """Whether the training feature is installed, the feature registry being the authority for this workspace."""
    from arena_cli import common

    return "training" in common._reg_list()


def _planner_availability_problems(contest: Contest) -> list[str]:
    """DRL-family contestants whose planner or the training feature is not installed, with the command that installs it."""
    from arena_planners import resolver

    try:
        available = resolver.list_available()
    except (OSError, RuntimeError) as exc:
        return [f"the planner registry cannot be read, run `arena feature planners check`: {exc}"]
    known = sorted(set(available["registry"]) | set(available["rosnav_rl"]))
    problems: list[str] = []
    training_available: bool | None = None
    for contestant in contest.contestants:
        for cap in ("mobile", "arm", "gripper", "lift"):
            driver = _contest_drivers(contestant.args, cap)
            if driver not in _DRL_DRIVERS:
                continue
            if training_available is None:
                training_available = _training_available()
            if not training_available:
                problems.append(f"{contestant.name}: {cap} driver {driver!r} needs rosnav_rl, run `arena feature training install`")
                continue
            planner = _cap_planner(contestant.args, cap)
            if planner is not None and not resolver.installed_in_registry(planner):
                problems.append(f"{contestant.name}: {cap} planner {planner!r} is not checked out, run `arena feature planners add {planner}` or pick one of {known}")
    return problems


def check_manifest(data: object) -> dict[str, Any]:
    from arena_evaluation.presentation.viz_manifest import VizManifest

    if not isinstance(data, dict):
        return {"valid": False, "problems": [f"a manifest is a mapping, got {type(data).__name__}"]}
    try:
        manifest = VizManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        return {"valid": False, "problems": [str(exc)]}
    for plot in manifest.plots:
        if not plot.data_key and not plot.uses and plot.data is None:
            return {"valid": False, "problems": [f"plot {plot.id!r} has no data_key, uses, or data specification"]}
    return {"valid": True, "n_plots": len(manifest.plots), "n_groups": len(manifest.groups), "problems": []}


def write_config(kind: str, name: str, text: str, *, overwrite: bool = False) -> pathlib.Path:
    """Write a suite, contest or manifest to the write target its resolver chain names, as a flat <name>.yaml."""
    stem = config_stem(name)
    if stem in config_names(kind) and not overwrite:
        raise ToolError(f"a {kind} named {stem!r} already exists", hint="pick another name or pass overwrite=true")
    target = asyncio.run(config_identifier(kind)(name=stem).resolve_write_path())
    path = target.parent / f"{target.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@domain.tool(read_only=True)
def evaluation_validate_suite(yaml_content: YamlText) -> dict[str, Any]:
    """Check a suite with the runner's Suite parser, then warn about task modes, config keys, maps, robots and models the registries do not know."""
    return check_suite(parse_yaml(yaml_content, "suite"))


@domain.tool(read_only=True)
def evaluation_validate_contest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a contest with the runner's Contest parser and its nav2 planner preflight, and its drivers against the registries."""
    return check_contest(parse_yaml(yaml_content, "contest"))


@domain.tool(read_only=True)
def evaluation_validate_manifest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a report manifest against the VizManifest schema."""
    return check_manifest(parse_yaml(yaml_content, "manifest"))


@domain.tool()
def evaluation_create_suite(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a suite (evaluation_config_schema and episode_task_modes give its keys) and write it where the runner resolves it by name."""
    verdict = check_suite(parse_yaml(yaml_content, "suite"))
    if not verdict["valid"]:
        raise ToolError("suite does not validate: " + "; ".join(verdict["problems"]), hint="evaluation_config_schema kind=suite shows the schema")
    return {"path": str(write_config("suite", name, yaml_content, overwrite=overwrite)), **verdict}


@domain.tool()
def evaluation_create_contest(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a contest (planners_list gives the planner names) and write it where the runner resolves it by name."""
    verdict = check_contest(parse_yaml(yaml_content, "contest"))
    if not verdict["valid"]:
        raise ToolError("contest does not validate: " + "; ".join(verdict["problems"]), hint="planners_list lists the planners")
    return {"path": str(write_config("contest", name, yaml_content, overwrite=overwrite)), **verdict}


@domain.tool()
def evaluation_create_manifest(
    name: str,
    *,
    yaml_content: Annotated[str | None, pydantic.Field(description="complete manifest YAML, or omit and give a template")] = None,
    template: Annotated[str | None, pydantic.Field(description="a resolvable manifest to start from (evaluation_manifests)")] = None,
    run_id: Annotated[str | None, pydantic.Field(description="write into this run's directory instead of the shared manifest root")] = None,
    overwrite: Overwrite = False,
) -> dict[str, Any]:
    """Write a report manifest, from YAML or a copy of a template, into a run directory or the shared manifest root (table notes: options.notes, options.rows, options.note, options.notes_key)."""
    import yaml

    from arena_evaluation.presentation.manifest_registry import resolve_manifest

    if template is not None and yaml_content is None:
        try:
            manifest = resolve_manifest(config_stem(template))
        except FileNotFoundError as exc:
            raise ToolError(str(exc), hint="evaluation_manifests lists the templates") from exc
        manifest.name = name
        yaml_content = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    elif yaml_content is None or template is not None:
        raise ToolError("pass yaml_content or template")
    verdict = check_manifest(parse_yaml(yaml_content, "manifest"))
    if not verdict["valid"]:
        raise ToolError("manifest does not validate: " + "; ".join(verdict["problems"]))
    if run_id is None:
        return {"path": str(write_config("manifest", name, yaml_content, overwrite=overwrite)), **verdict}
    path = run_path(run_id) / f"{config_stem(name)}.yaml"
    if path.exists() and not overwrite:
        raise ToolError(f"{path} exists", hint="pick another name or pass overwrite=true")
    path.write_text(yaml_content)
    return {"path": str(path), **verdict}
