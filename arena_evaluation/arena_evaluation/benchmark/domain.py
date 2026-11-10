"""ROS 2 domain selection for a benchmark run.

The runner starts the arena runtime and its envs as child processes, so the domain has to be in the
process environment before either starts: a launch argument only reaches the ROS graph. A domain that
already serves ``/arena/register_env`` has another arena runtime on it (another session or another
user sharing the host network), and a second runtime on the same domain refuses to start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import typing
from dataclasses import dataclass

ROS_DOMAIN_ID_ENV: typing.Final[str] = "ROS_DOMAIN_ID"
GZ_PARTITION_ENV: typing.Final[str] = "GZ_PARTITION"
REGISTER_ENV_SERVICE: typing.Final[str] = "/arena/register_env"
# ROS 2 accepts 0-232; 0 is the ROS default and is therefore the most crowded.
DOMAIN_CANDIDATES: typing.Final[tuple[int, ...]] = (*range(1, 233), 0)
_PROBE_TIMEOUT_S: typing.Final[float] = 0.4
_PROBE_SCRIPT: typing.Final[str] = """
import sys

import rclpy
from arena_runtime_msgs.srv import SpawnEnv

rclpy.init()
node = rclpy.create_node("arena_evaluation_domain_probe")
try:
    client = node.create_client(SpawnEnv, sys.argv[1])
    present = client.wait_for_service(timeout_sec=float(sys.argv[2]))
finally:
    node.destroy_node()
    rclpy.shutdown()
sys.exit(0 if present else 3)
"""
_PROBE_ABSENT_EXIT: typing.Final[int] = 3


DomainSource = typing.Literal["explicit", "inherited", "resume", "probed"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainChoice:
    """The ROS domain a run uses and where the choice came from."""

    domain_id: int
    source: DomainSource
    probed: tuple[int, ...]

    @property
    def explicit(self) -> bool:
        return self.source == "explicit"


class DomainBusyError(RuntimeError):
    """Raised when an explicitly requested ROS domain already has an arena runtime on it."""


def runtime_on_domain(domain_id: int, *, timeout_s: float = _PROBE_TIMEOUT_S) -> bool:
    """Whether ``/arena/register_env`` is already served on this ROS 2 domain.

    The probe runs in a child process because the ROS domain is read from the environment once, when
    the rmw context initialises, and a process cannot initialise a second context on another domain.
    """
    env = {**os.environ, ROS_DOMAIN_ID_ENV: str(domain_id)}
    done = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, REGISTER_ENV_SERVICE, str(timeout_s)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s + 30.0,
        check=False,
    )
    if done.returncode == 0:
        return True
    if done.returncode == _PROBE_ABSENT_EXIT:
        return False
    raise RuntimeError(f"the ROS domain probe for domain {domain_id} failed with exit code {done.returncode}; the ROS environment is not usable")


def choose_domain(
    explicit: int | None = None,
    *,
    probe: typing.Callable[[int], bool] = runtime_on_domain,
    candidates: typing.Iterable[int] = DOMAIN_CANDIDATES,
) -> DomainChoice:
    """The run's ROS domain: the explicit one when free, else the lowest candidate with no runtime on it."""
    if explicit is not None:
        if probe(explicit):
            raise DomainBusyError(
                f"ROS domain {explicit} already has an arena runtime (another session shares this host network); "
                f"pass a free ros_domain_id=, shut that runtime down, or let the runner pick one by omitting it"
            )
        return DomainChoice(domain_id=explicit, source="explicit", probed=(explicit,))

    probed: list[int] = []
    for candidate in candidates:
        if not probe(candidate):
            return DomainChoice(domain_id=candidate, source="probed", probed=tuple(probed))
        probed.append(candidate)
    raise DomainBusyError(
        f"every candidate ROS domain already has an arena runtime ({len(probed)} checked, e.g. {probed[:8]}); "
        f"shut one down or pass an explicit ros_domain_id="
    )


def gz_partition(domain_id: int) -> str:
    """Gazebo transport partition matching the ROS domain, so simulators on other domains cannot see each other."""
    return f"arena_{domain_id}"


def apply_domain(choice: DomainChoice) -> None:
    """Put the chosen domain and its Gazebo partition into this process environment for every child it starts."""
    os.environ[ROS_DOMAIN_ID_ENV] = str(choice.domain_id)
    os.environ[GZ_PARTITION_ENV] = gz_partition(choice.domain_id)


_DESCRIPTIONS: typing.Final[dict[str, str]] = {
    "explicit": "requested",
    "inherited": "from ROS_DOMAIN_ID in the environment",
    "resume": "reused from the run manifest",
    "probed": "picked, the lowest domain with no arena runtime",
}


def describe(choice: DomainChoice) -> str:
    """One line for the run log and console."""
    return f"benchmark: ROS domain {choice.domain_id} ({_DESCRIPTIONS[choice.source]}, GZ_PARTITION={gz_partition(choice.domain_id)})"