import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import json
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree  # FIX: spatial index for visibility
from shapely.ops import nearest_points


@dataclass
class GoalTask:
    goal_id: str
    cp_id: str
    goal_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    cp_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    exit_waypoints: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class AgentPlan:
    agent_name: PlayerName
    tasks: List[GoalTask]


class GlobalPlanMessage(BaseModel):
    agent_plans: Dict[str, AgentPlan]
    goal_positions: Dict[str, Tuple[float, float]]
    cp_positions: Dict[str, Tuple[float, float]]
    num_agents: int


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """PDM4AR agent keeping a global plan and returning placeholder commands."""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        self.name = ""
        self.params = Pdm4arAgentParams()
        self.plan: List[GoalTask] = []
        self.all_agent_plans: Dict[str, List[GoalTask]] = {}
        self.current_task_idx: int = 0
        self.k_v: float = 1.0
        self.k_w: float = 2.0
        self.phase: str = "goal"
        self.goal_positions: Dict[str, Tuple[float, float]] = {}
        self.cp_positions: Dict[str, Tuple[float, float]] = {}
        self.current_wp_idx: int = 0
        self.avoid_radius: float = 0.7
        self.k_avoid: float = 2.0
        # --- nouveaux attributs pour le contrôle continu ---
        self.current_target: Tuple[float, float] | None = None  # cible XY courante
        self.last_replan_time: float = 0.0
        self.replan_period: float = 0.5  # re-planning toutes les 0.5s (~2 Hz)
        self.goal_tol: float = 0.28  # tolérance pour considérer goal/CP atteint
        # --- état pour le suivi continu du chemin (path tracking) ---
        self.current_path: list[tuple[float, float]] = []  # polyline actuelle (goal ou CP)
        self.path_progress_idx: int = 0  # index de progression le long du chemin
        self.lookahead_dist: float = 0.8  # distance de lookahead pour pure pursuit
        self.min_path_rebuild_period: float = 0.5  # on ne reconstruit pas le chemin trop souvent
        self.reverse_mode: bool = False
        self._last_reverse_log_time: float = -1e9
        self.num_agents: int = 0
        self.goals_collected_count: Dict[str, int] = {}
        self._last_collected_goal_id: Dict[str, str | None] = {}
        self.cp_priority_lock: Dict[Tuple[str, str], bool] = {}
        self.detour_side_lock: Dict[str, str] = {}
        self.detour_lock_time: Dict[str, float] = {}
        self._blocked_by_dynamic: bool = False
        self.yielding: bool = False
        self.yield_cp_id: str | None = None
        self.yield_to_agent: PlayerName | None = None
        self.yield_enter_radius_mult: float = 6.0
        self.yield_exit_radius_mult: float = 7.0
        self.detour_active: bool = False
        self.detour_end_idx: int | None = None
        self.detour_start_time: float | None = None
        self.detour_timeout: float = 3.0
        self.early_rejoin_hits: int = 0
        self.early_rejoin_required_hits: int = 3
        self.early_rejoin_last_min_dist: float | None = None
        self.early_rejoin_cooldown_until: float = 0.0
        self.last_global_rebuild_time: float = -1e9
        self.last_detour_time: float = -1e9
        self._stop_log_path: str | None = None
        self._last_stop_line: str | None = None
        self._last_stop_time: float = -1e9
        self._stop_log_min_period: float = 0.5
        self._last_pos: Tuple[float, float] | None = None
        self._last_dist_to_cp: Dict[str, float] = {}
        self._last_dist_to_agent: Dict[PlayerName, float] = {}
        self._last_dist_to_yield: Dict[PlayerName, float] = {}
        self._yield_recede_count: Dict[PlayerName, int] = {}
        self._yield_recede_latched: Dict[PlayerName, bool] = {}
        self.dans_couloir: bool = False
        self.couloir_devant: bool = False
        self._last_couloir_entry_sample: tuple[int, float] | None = None
        self.couloir_backoff_active: bool = False
        self.couloir_backoff_target: tuple[float, float] | None = None
        self.couloir_wait_exit: bool = False
        self.couloir_yield_to: str | None = None
        self.couloir_entry_latched: tuple[float, float, float, float] | None = None
        self.couloir_ignore_until: float = 0.0
        self.couloir_ignore_other = None  # type: Optional[PlayerName]
        self.corridor_lock: Dict[str, bool] = {}
        self.corridor_lock_active: set[str] = set()

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.name = init_sim_obs.my_name
        self.goal = init_sim_obs.goal
        self.static_obstacles = list(init_sim_obs.dg_scenario.static_obstacles)
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.plan = []
        self.current_task_idx = 0
        self.phase = "goal"
        self.goal_positions = {}
        self.cp_positions = {}
        self.current_wp_idx = 0
        self.current_target = None
        self.last_replan_time = 0.0
        self.current_path = []
        self.path_progress_idx = 0
        self.num_agents = 0
        self.goals_collected_count = {}
        self._last_collected_goal_id = {}
        self.all_agent_plans = {}
        self.cp_priority_lock = {}
        self.detour_side_lock = {}
        self.detour_lock_time = {}
        self._blocked_by_dynamic = False
        self.yielding = False
        self.yield_cp_id = None
        self.yield_to_agent = None
        self.detour_active = False
        self.detour_end_idx = None
        self.detour_start_time = None
        self.early_rejoin_hits = 0
        self.early_rejoin_required_hits = 3
        self.early_rejoin_last_min_dist = None
        self.early_rejoin_cooldown_until = 0.0
        self.reverse_mode = False
        self._last_reverse_log_time = -1e9
        self.last_global_rebuild_time = -1e9
        self.last_detour_time = -1e9
        self._stop_log_path = f"stop_events_{self.name}.txt"
        self._last_stop_line = None
        self._last_stop_time = -1e9
        self.dans_couloir = False
        self.couloir_devant = False
        self.couloir_backoff_active = False
        self.couloir_backoff_target = None
        self.couloir_wait_exit = False
        self.couloir_yield_to = None
        self.couloir_entry_latched = None
        self.couloir_ignore_until = 0.0
        self.couloir_ignore_other = None
        self.couloir_ignore_until = 0.0
        self.couloir_ignore_other = None
        self._last_couloir_entry_sample = None
        self.corridor_lock = {}
        self.corridor_lock_active = set()
        self._last_pos = None
        self._last_dist_to_cp = {}
        self._last_dist_to_agent = {}
        self._last_dist_to_yield = {}
        self._yield_recede_count = {}
        self._yield_recede_latched = {}

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        self.all_agent_plans = {str(k): list(v.tasks) for k, v in global_plan.agent_plans.items()}
        print("agent on_receive_global_plan:", self.name, "available plans:", list(global_plan.agent_plans.keys()))
        agent_plan = global_plan.agent_plans.get(str(self.name))
        if agent_plan is not None:
            self.plan = list(agent_plan.tasks)
            self.current_task_idx = 0
        else:
            self.plan = []
            self.current_task_idx = 0
        self.num_agents = global_plan.num_agents
        print(self.num_agents)

        if self.num_agents <= 1:
            # CAS SOLO : Je suis seul (ou le message est vide)
            self.lookahead_dist = 1
        else:
            # CAS MULTI : Il y a au moins un autre agent avec moi
            self.lookahead_dist = 0.8
        self.phase = "goal"
        self.goal_positions = dict(global_plan.goal_positions)
        self.cp_positions = dict(global_plan.cp_positions)
        self.current_wp_idx = 0
        self.current_target = None
        self.last_replan_time = 0.0
        self.current_path = []
        self.path_progress_idx = 0
        self.num_agents = global_plan.num_agents
        self.goals_collected_count = {}
        self._last_collected_goal_id = {}
        self.cp_priority_lock = {}
        self.detour_side_lock = {}
        self.detour_lock_time = {}
        self._blocked_by_dynamic = False
        self.yielding = False
        self.yield_cp_id = None
        self.yield_to_agent = None
        self.detour_active = False
        self.detour_end_idx = None
        self.detour_start_time = None
        self.reverse_mode = False
        self._last_reverse_log_time = -1e9
        self.last_global_rebuild_time = -1e9
        self.last_detour_time = -1e9
        self._stop_log_path = f"stop_events_{self.name}.txt"
        self._last_stop_line = None
        self._last_stop_time = -1e9
        self.dans_couloir = False
        self.couloir_devant = False
        self.couloir_backoff_active = False
        self.couloir_backoff_target = None
        self.couloir_wait_exit = False
        self.couloir_yield_to = None
        self.couloir_entry_latched = None

    def _build_current_path(self, task: GoalTask) -> None:
        """
        Construit self.current_path pour la tâche et la phase en cours.
        - Si phase == "goal": on utilise task.goal_waypoints s'ils existent,
          sinon on met simplement le centre du goal (self.goal_positions[task.goal_id]).
        - Si phase == "cp": on utilise task.cp_waypoints s'ils existent,
          sinon le centre du CP (self.cp_positions[task.cp_id]).
        On remet path_progress_idx à 0.
        """
        path: list[tuple[float, float]] = []

        if self.phase == "goal":
            if task.goal_waypoints:
                path = [(float(x), float(y)) for (x, y) in task.goal_waypoints]
            elif task.goal_id in self.goal_positions:
                path = [self.goal_positions[task.goal_id]]
        elif self.phase == "cp":
            if task.cp_waypoints:
                path = [(float(x), float(y)) for (x, y) in task.cp_waypoints]
            elif task.cp_id in self.cp_positions:
                path = [self.cp_positions[task.cp_id]]
        elif self.phase == "exit":
            if task.exit_waypoints:
                path = [(float(x), float(y)) for (x, y) in task.exit_waypoints]
            elif task.cp_id in self.cp_positions:
                path = [self.cp_positions[task.cp_id]]

        self.current_path = path
        self.path_progress_idx = 0

    def _update_continuous_target(self, sim_obs: SimObservations) -> None:
        """
        Met à jour self.current_target en suivant le chemin global (self.current_path)
        de manière continue (path tracking style pure pursuit).
        - On reconstruit self.current_path périodiquement ou lorsqu'elle est vide.
        - On projette la position du robot sur la polyline.
        - On choisit un point de lookahead à distance self.lookahead_dist le long du chemin.
        """
        if not self.plan or self.current_task_idx >= len(self.plan):
            self.reverse_mode = False
            self.current_target = None
            return

        # temps de simulation (si disponible)
        sim_time = self._timef(getattr(sim_obs, "time", None))

        task = self.plan[self.current_task_idx]

        if (
            self.detour_active
            and self.detour_start_time is not None
            and sim_time - self.detour_start_time > self.detour_timeout
        ):
            print(f"DETOUR TIMEOUT @ {sim_time:.2f}, aborting")
            self._end_detour(sim_time, reason="timeout")

        need_rebuild = not self.current_path or (
            not self.detour_active and sim_time - self.last_global_rebuild_time > self.min_path_rebuild_period
        )
        if need_rebuild:
            if self.detour_active and self.current_path:
                print("SKIP REBUILD (detour_active)")
            else:
                self._build_current_path(task)
                self.last_global_rebuild_time = float(sim_time)
                self.last_replan_time = float(sim_time)

        if not self.current_path:
            self.reverse_mode = False
            self.current_target = None
            return

        state = sim_obs.players[self.name].state
        px, py = float(state.x), float(state.y)
        psi = float(state["psi"]) if isinstance(state, Mapping) else float(state.psi)

        path = self.current_path
        n = len(path)
        if n == 1:
            self.reverse_mode = False
            self.current_target = path[0]
            return

        search_back = 15
        search_forward = 40
        base_idx = min(self.path_progress_idx, n - 2)
        start_idx = max(0, base_idx - search_back)
        end_idx = min(n - 2, base_idx + search_forward)
        if end_idx < start_idx:
            start_idx = end_idx

        fallback_idx = max(0, min(base_idx, n - 2))
        fallback_next = fallback_idx + 1
        best_dir_x = path[fallback_next][0] - path[fallback_idx][0]
        best_dir_y = path[fallback_next][1] - path[fallback_idx][1]
        best_idx = fallback_idx
        best_t = 0.0
        best_dist = float("inf")

        for i in range(start_idx, end_idx + 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            vx = x2 - x1
            vy = y2 - y1
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-8:
                continue

            wx = px - x1
            wy = py - y1
            t = (wx * vx + wy * vy) / seg_len2
            t_clamped = max(0.0, min(1.0, t))
            proj_x = x1 + t_clamped * vx
            proj_y = y1 + t_clamped * vy

            d2 = (proj_x - px) ** 2 + (proj_y - py) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_idx = i
                best_t = t_clamped
                best_dir_x = vx
                best_dir_y = vy

        dir_norm = math.hypot(best_dir_x, best_dir_y)
        if dir_norm < 1e-6:
            path_heading = psi
        else:
            path_heading = math.atan2(best_dir_y, best_dir_x)

        angle_diff = self._wrap_to_pi(path_heading - psi)
        self.reverse_mode = abs(angle_diff) > (math.pi / 2)

        progress_idx = max(self.path_progress_idx, best_idx)
        progress_idx = min(progress_idx, n - 2)
        self.path_progress_idx = progress_idx
        self._maybe_end_detour(sim_time, px, py)

        lookahead_remaining = self.lookahead_dist
        target_s = float(best_idx + best_t)
        if target_s < float(progress_idx):
            target_s = float(progress_idx)

        current_s_int = min(int(target_s), n - 2)
        t_fraction = target_s - current_s_int
        if current_s_int >= n - 1:
            self.current_target = path[-1]
            return

        # point de départ sur le segment courant
        x_start, y_start = path[current_s_int]
        x_next, y_next = path[current_s_int + 1]
        vx = x_next - x_start
        vy = y_next - y_start
        seg_len = float(np.hypot(vx, vy))
        if seg_len < 1e-6:
            self.current_target = path[-1]
            return

        # position actuelle projetée sur le segment
        start_offset = t_fraction * seg_len
        # distance restante sur le segment courant après la projection
        seg_remain = max(0.0, seg_len - start_offset)

        # on avance le long du chemin jusqu'à consommer lookahead_remaining
        idx = current_s_int
        offset_along_seg = start_offset

        while lookahead_remaining > seg_remain and idx < n - 2:
            lookahead_remaining -= seg_remain
            idx += 1
            x_start, y_start = path[idx]
            x_next, y_next = path[idx + 1]
            vx = x_next - x_start
            vy = y_next - y_start
            seg_len = float(np.hypot(vx, vy))
            if seg_len < 1e-6:
                seg_remain = 0.0
                continue
            offset_along_seg = 0.0
            seg_remain = seg_len

        if seg_remain < 1e-6:
            # plus vraiment de chemin -> viser le dernier point
            self.current_target = path[-1]
            return

        # cible sur le segment [idx, idx+1]
        t_look = (offset_along_seg + lookahead_remaining) / max(seg_len, 1e-6)
        t_look = max(0.0, min(1.0, t_look))
        x_target = x_start + t_look * (x_next - x_start)
        y_target = y_start + t_look * (y_next - y_start)

        self.current_target = (float(x_target), float(y_target))

    def _segment_intersects_disk(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        r: float,
    ) -> bool:
        ax, ay = a
        bx, by = b
        cx, cy = c
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-8:
            dist2 = (ax - cx) ** 2 + (ay - cy) ** 2
            return dist2 <= r * r
        t = ((cx - ax) * vx + (cy - ay) * vy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))
        px = ax + vx * t_clamped
        py = ay + vy * t_clamped
        dist2 = (px - cx) ** 2 + (py - cy) ** 2
        return dist2 <= (r + 1e-8) ** 2

    def _polyline_first_intersection_idx(
        self,
        path: list[tuple[float, float]],
        start_idx: int,
        c: tuple[float, float],
        r: float,
        max_segments: int = 8,
    ) -> int | None:
        n = len(path)
        if n < 2 or start_idx >= n - 1:
            return None
        end_idx = min(n - 1, start_idx + max_segments)
        for i in range(start_idx, end_idx):
            if self._segment_intersects_disk(path[i], path[i + 1], c, r):
                return i
        return None

    def _is_other_on_my_path(
        self,
        other_xy: tuple[float, float],
        radius: float,
        max_segments: int = 12,
    ) -> bool:
        path = self.current_path
        if not path or len(path) < 2:
            return False
        start_idx = int(getattr(self, "path_progress_idx", 0))
        hit = self._polyline_first_intersection_idx(path, start_idx, other_xy, float(radius), max_segments=max_segments)
        return hit is not None

    def _am_i_priority_vs(self, other_name: str) -> bool:
        return str(self.name) < str(other_name)

    def _choose_exit_idx(
        self,
        path: list[tuple[float, float]],
        entry_idx: int,
        c: tuple[float, float],
        r: float,
        max_lookahead: int = 6,
    ) -> int:
        n = len(path)
        if n < 2:
            return 0
        entry_idx = max(0, min(entry_idx, n - 2))
        start_idx = entry_idx + 1
        end_idx = min(n - 1, start_idx + max_lookahead)
        for idx in range(start_idx, end_idx + 1):
            px, py = path[idx]
            if (px - c[0]) ** 2 + (py - c[1]) ** 2 <= r * r:
                continue
            if idx < n - 1 and self._segment_intersects_disk(path[idx], path[idx + 1], c, r):
                continue
            return idx
        return min(entry_idx + 3, n - 1)

    def _compute_local_detour(
        self,
        S: tuple[float, float],
        T: tuple[float, float],
        c: tuple[float, float],
        r: float,
        static_obstacles: Sequence[StaticObstacle],
        n_samples: int = 16,
        side: str | None = None,
    ) -> list[tuple[float, float]] | None:
        Sx, Sy = S
        Tx, Ty = T
        cx, cy = c
        direction_x = Tx - Sx
        direction_y = Ty - Sy
        dir_norm = float(np.hypot(direction_x, direction_y))
        angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        base_radius = r + 0.15
        radii = [base_radius, base_radius + 0.2]
        samples: list[tuple[float, float]] = []
        for radius in radii:
            for angle in angles:
                px = cx + radius * float(np.cos(angle))
                py = cy + radius * float(np.sin(angle))
                samples.append((px, py))

        sample_candidates = samples
        if side in {"left", "right"} and dir_norm > 1e-6:
            filtered: list[tuple[float, float]] = []
            for px, py in samples:
                cross = direction_x * (py - cy) - direction_y * (px - cx)
                if side == "left" and cross >= 0.0:
                    filtered.append((px, py))
                elif side == "right" and cross <= 0.0:
                    filtered.append((px, py))
            min_samples = max(4, n_samples // 2)
            if len(filtered) >= min_samples:
                sample_candidates = filtered

        nodes = [(Sx, Sy), (Tx, Ty)] + sample_candidates
        if len(nodes) < 2:
            return None

        buffer_margin = float(self.sg.radius) + 0.05
        buffered_obstacles = []
        for obs in static_obstacles:
            try:
                buffered_obstacles.append(obs.shape.buffer(buffer_margin))
            except Exception:
                continue

        r_buffer = r + 0.05
        graph: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(nodes))}
        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if self._segment_intersects_disk(a, b, c, r_buffer):
                    continue
                line = LineString([a, b])
                blocked = False
                for buf in buffered_obstacles:
                    if line.intersects(buf):
                        blocked = True
                        break
                if blocked:
                    continue
                edge_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                if edge_len < 1e-6:
                    continue
                graph[i].append((j, edge_len))
                graph[j].append((i, edge_len))

        num_nodes = len(nodes)
        dist = [float("inf")] * num_nodes
        prev: list[int | None] = [None] * num_nodes
        visited = [False] * num_nodes
        dist[0] = 0.0

        for _ in range(num_nodes):
            u = -1
            best = float("inf")
            for idx in range(num_nodes):
                if not visited[idx] and dist[idx] < best:
                    best = dist[idx]
                    u = idx
            if u == -1:
                break
            if u == 1:
                break
            visited[u] = True
            for v, weight in graph.get(u, []):
                if visited[v]:
                    continue
                nd = dist[u] + weight
                if nd + 1e-9 < dist[v]:
                    dist[v] = nd
                    prev[v] = u

        if not np.isfinite(dist[1]):
            return None

        path_indices: list[int] = []
        idx = 1
        while idx is not None:
            path_indices.append(idx)
            idx = prev[idx]
        path_indices.reverse()

        return [(float(nodes[i][0]), float(nodes[i][1])) for i in path_indices]

    def _splice_detour_into_current_path(
        self,
        detour: list[tuple[float, float]],
        exit_idx: int,
        sim_time: float,
    ) -> None:
        if not detour:
            return
        if not self.current_path:
            self.current_path = list(detour)
            self.path_progress_idx = 0
            self.last_replan_time = float(sim_time)
            return
        exit_idx = max(0, min(exit_idx, len(self.current_path) - 1))
        prefix = detour[:-1]
        suffix = self.current_path[exit_idx:]
        detour_last_idx = len(prefix)
        new_path = prefix + suffix
        if not new_path:
            new_path = detour
        self.current_path = new_path
        self.path_progress_idx = 0
        self._start_detour(sim_time, detour_last_idx)
        self.last_replan_time = float(sim_time)

    def _prune_detour_side_locks(self, sim_time: float) -> None:
        expired: list[str] = []
        for key, lock_time in self.detour_lock_time.items():
            if sim_time - lock_time > 2.0:
                expired.append(key)
        for key in expired:
            self.detour_side_lock.pop(key, None)
            self.detour_lock_time.pop(key, None)

    def _apply_dynamic_detour_if_needed(
        self,
        robot_pos: tuple[float, float],
        sim_time: float | None,
        other_players: Mapping[PlayerName, SimObservations],
    ) -> None:
        sim_time = self._timef(sim_time)
        last_replan = float(self.last_replan_time)
        self._prune_detour_side_locks(sim_time)

        path = self.current_path
        if not path or len(path) < 2:
            self._blocked_by_dynamic = False
            return

        blocked_needed = False
        detour_applied = False
        S = robot_pos
        R = float(self.sg.radius)
        trigger_r = 4.0 * R
        avoid_r = 2.0 * R + 0.2

        for other_name, other_obs in other_players.items():
            if detour_applied:
                break
            other_state = other_obs.state
            ox = other_state["x"] if isinstance(other_state, Mapping) else other_state.x
            oy = other_state["y"] if isinstance(other_state, Mapping) else other_state.y
            c = (float(ox), float(oy))

            entry_idx = self._polyline_first_intersection_idx(
                path,
                self.path_progress_idx,
                c,
                trigger_r,
                max_segments=8,
            )
            if entry_idx is None:
                continue
            blocked_needed = True

            if sim_time - last_replan < self.replan_period:
                continue

            first_exit = self._choose_exit_idx(path, entry_idx, c, avoid_r, max_lookahead=6)
            first_exit = max(0, min(first_exit, len(path) - 1))
            T = path[first_exit]

            key = str(other_name)
            side = self.detour_side_lock.get(key)
            lock_time = self.detour_lock_time.get(key, -1e9)
            direction_x = T[0] - S[0]
            direction_y = T[1] - S[1]
            if side is None or sim_time - lock_time > 1.5:
                if np.hypot(direction_x, direction_y) >= 1e-6:
                    to_center_x = c[0] - S[0]
                    to_center_y = c[1] - S[1]
                    cross = direction_x * to_center_y - direction_y * to_center_x
                    side = "left" if cross >= 0.0 else "right"
                else:
                    side = "left"
                self.detour_side_lock[key] = side
            self.detour_lock_time[key] = sim_time

            MAX_EXIT_TRIES = 6
            for k in range(MAX_EXIT_TRIES):
                cand_exit = min(first_exit + k, len(path) - 1)
                T = path[cand_exit]
                detour = self._compute_local_detour(
                    S,
                    T,
                    c,
                    avoid_r,
                    self.static_obstacles,
                    n_samples=16,
                    side=side,
                )
                if detour and len(detour) >= 2:
                    self._splice_detour_into_current_path(detour, cand_exit, sim_time)
                    detour_applied = True
                    break

        self._blocked_by_dynamic = blocked_needed and not detour_applied

    def _has_cp_priority_over(
        self,
        sim_obs: SimObservations,
        other_name: str,
        cp_id_key: str,
        dist_me: float,
        dist_other: float,
    ) -> bool:
        """Détermine la priorité autour d'un CP en suivant les règles spécifiques."""
        my_obs = sim_obs.players.get(self.name)
        other_obs = sim_obs.players.get(other_name)
        if my_obs is None or other_obs is None:
            return True
        _ = cp_id_key  # cp-specific logic may require the cp id for future diagnostics

        def _get_collected_goal(obs):
            carried = getattr(obs, "collected_goal_id", None)
            if carried is None and isinstance(obs, Mapping):
                carried = obs.get("collected_goal_id")
            return carried

        my_carried = _get_collected_goal(my_obs)
        other_carried = _get_collected_goal(other_obs)
        me_carry = bool(my_carried)
        other_carry = bool(other_carried)
        my_name = str(self.name)
        other_str = str(other_name)
        tie = abs(dist_me - dist_other) < 1e-6

        if me_carry != other_carry:
            return not me_carry

        if me_carry and other_carry:
            if tie:
                return my_name < other_str
            return dist_me < dist_other

        # Aucun ne transporte
        if tie:
            return my_name < other_str
        return dist_me > dist_other

    def _prune_cp_priority_locks(self, sim_obs: SimObservations) -> None:
        """Supprime les verrous de priorité pour les agents qui ne sont plus observés
        OU quand les agents sont suffisamment espacés (>= 5R)."""
        active_players = {str(name) for name in sim_obs.players.keys()}

        # Position de self
        my_obs = sim_obs.players.get(self.name)
        if my_obs is None:
            self.cp_priority_lock.clear()
            return

        my_st = my_obs.state
        mx = float(my_st["x"]) if isinstance(my_st, Mapping) else float(my_st.x)
        my = float(my_st["y"]) if isinstance(my_st, Mapping) else float(my_st.y)

        R = float(self.sg.radius)
        sep_thresh = 5.0 * R

        for lock_key in list(self.cp_priority_lock.keys()):
            other_name, _ = lock_key
            # purge si plus observé
            if other_name not in active_players:
                del self.cp_priority_lock[lock_key]
                continue

            other_obs = sim_obs.players.get(other_name)
            if other_obs is None:
                del self.cp_priority_lock[lock_key]
                continue

            ost = other_obs.state
            ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
            oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)

            if float(np.hypot(mx - ox, my - oy)) >= sep_thresh:
                del self.cp_priority_lock[lock_key]

    def _is_converging_to_point(
        self,
        sim_obs: SimObservations,
        point_xy: tuple[float, float],
        key: str,
        dist_eps: float = 1e-3,
        max_angle_deg: float = 80.0,
    ) -> bool:
        """Retourne True si la distance vers le point diminue ou si l'orientation pointe vers le point."""
        st = sim_obs.players[self.name].state
        x = float(st["x"]) if isinstance(st, Mapping) else float(st.x)
        y = float(st["y"]) if isinstance(st, Mapping) else float(st.y)
        psi = float(st["psi"]) if isinstance(st, Mapping) else float(st.psi)

        dx = point_xy[0] - x
        dy = point_xy[1] - y
        d = (dx * dx + dy * dy) ** 0.5

        last_d = self._last_dist_to_cp.get(key)
        self._last_dist_to_cp[key] = d
        dist_decreasing = (last_d is not None) and (d < last_d - dist_eps)

        ang_to_target = math.atan2(dy, dx)
        ang_err = (ang_to_target - psi + math.pi) % (2 * math.pi) - math.pi
        angle_ok = abs(ang_err) <= math.radians(max_angle_deg)

        return dist_decreasing or angle_ok

    def _is_converging_to_agent(
        self,
        sim_obs: SimObservations,
        other_name: PlayerName,
        dist_eps: float = 1e-3,
    ) -> bool:
        """Retourne True si la distance vers un autre agent diminue."""
        st = sim_obs.players[self.name].state
        x = float(st["x"]) if isinstance(st, Mapping) else float(st.x)
        y = float(st["y"]) if isinstance(st, Mapping) else float(st.y)
        ost = sim_obs.players[other_name].state
        ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
        oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)

        dx = ox - x
        dy = oy - y
        d = (dx * dx + dy * dy) ** 0.5

        last_d = self._last_dist_to_agent.get(other_name)
        self._last_dist_to_agent[other_name] = d
        return (last_d is not None) and (d < last_d - dist_eps)

    def _distance_libre_le_long(
        self,
        origin_x: float,
        origin_y: float,
        direction: tuple[float, float],
        max_dist: float = 2.0,
        pas: float = 0.2,
    ) -> float:
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return float(max_dist)
        dx /= norm
        dy /= norm
        max_d = float(max_dist)
        step = float(pas) if pas > 0.0 else 0.2
        marge = 0.5 * float(self.sg.radius)
        dist = 0.0
        obstacles = self.static_obstacles or ()
        while dist <= max_d + 1e-9:
            probe = Point(origin_x + dx * dist, origin_y + dy * dist)
            for obs in obstacles:
                shape = getattr(obs, "shape", None)
                if shape is None:
                    continue
                try:
                    if probe.distance(shape) <= marge:
                        return dist
                except Exception:
                    continue
            dist += step
        return max_d

    def _detecter_couloir_local(self, state: object) -> bool:
        try:
            x = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
            y = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
            psi = float(state["psi"]) if isinstance(state, Mapping) else float(state.psi)
        except (KeyError, TypeError, AttributeError, ValueError):
            return False
        fx = math.cos(psi)
        fy = math.sin(psi)
        if self.reverse_mode:
            fx = -fx
            fy = -fy
        avant = (fx, fy)
        gauche = (-fy, fx)
        droite = (-gauche[0], -gauche[1])
        d_gauche = self._distance_libre_le_long(x, y, gauche, max_dist=2.0, pas=0.2)
        d_droite = self._distance_libre_le_long(x, y, droite, max_dist=2.0, pas=0.2)
        d_avant = self._distance_libre_le_long(x, y, avant, max_dist=3.0, pas=0.2)
        largeur = d_gauche + d_droite
        diam_robot = 2.0 * float(self.sg.radius)
        return (largeur < (diam_robot + 0.8)) and (d_avant > largeur)

    def _point_dans_couloir(
        self,
        x: float,
        y: float,
        direction: tuple[float, float],
    ) -> bool:
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return False
        tx = dx / norm
        ty = dy / norm
        normal_gauche = (ty, -tx)
        normal_droite = (-normal_gauche[0], -normal_gauche[1])
        d_gauche = self._distance_libre_le_long(x, y, normal_gauche, max_dist=2.0, pas=0.2)
        d_droite = self._distance_libre_le_long(x, y, normal_droite, max_dist=2.0, pas=0.2)
        d_avant = self._distance_libre_le_long(x, y, (tx, ty), max_dist=3.0, pas=0.2)
        largeur = d_gauche + d_droite
        diam_robot = 2.0 * float(self.sg.radius)
        return (largeur < (diam_robot + 0.8)) and (d_avant > largeur)

    def _va_entrer_dans_couloir(
        self,
        state: object,
        distance_lookahead: float = 2.5,
        pas_echant: float = 0.5,
    ) -> bool:
        if not self.current_path or len(self.current_path) < 2:
            return False
        try:
            px = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
            py = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
        except (KeyError, TypeError, AttributeError, ValueError):
            return False
        lookahead = float(distance_lookahead)
        if lookahead <= 0.0:
            return False
        step = float(pas_echant) if pas_echant > 0.0 else 0.1
        path = self.current_path
        best_idx = 0
        best_dist = math.inf
        for idx, (px_path, py_path) in enumerate(path):
            dist_robot = math.hypot(px_path - px, py_path - py)
            if dist_robot < best_dist:
                best_dist = dist_robot
                best_idx = idx
        start_idx = max(0, best_idx - 1)
        if start_idx >= len(path) - 1:
            return False
        parcourus = 0.0
        for seg_idx in range(start_idx, len(path) - 1):
            x0, y0 = path[seg_idx]
            x1, y1 = path[seg_idx + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            ux = dx / seg_len
            uy = dy / seg_len
            s = 0.0
            last_s = -1.0
            while s <= seg_len and (parcourus + s) <= lookahead:
                sample_x = x0 + ux * s
                sample_y = y0 + uy * s
                if self._point_dans_couloir(sample_x, sample_y, (ux, uy)):
                    return True
                last_s = s
                s += step
            if (parcourus + seg_len) <= lookahead and not math.isclose(last_s, seg_len, abs_tol=1e-6):
                if self._point_dans_couloir(x1, y1, (ux, uy)):
                    return True
            parcourus += seg_len
            if parcourus >= lookahead:
                break
        return False

    def _is_yield_other_receding(
        self,
        sim_obs: SimObservations,
        other_name: PlayerName,
        dist_eps: float = 0.02,
        n_confirm: int = 3,
    ) -> bool:
        if other_name is None:
            return False
        other_obs = sim_obs.players.get(other_name)
        if other_obs is None:
            self._yield_recede_count.pop(other_name, None)
            self._last_dist_to_yield.pop(other_name, None)
            self._yield_recede_latched.pop(other_name, None)
            return False

        my_obs = sim_obs.players.get(self.name)
        if my_obs is None:
            return False

        try:
            sx = float(my_obs.state["x"]) if isinstance(my_obs.state, Mapping) else float(my_obs.state.x)
            sy = float(my_obs.state["y"]) if isinstance(my_obs.state, Mapping) else float(my_obs.state.y)
            ox = float(other_obs.state["x"]) if isinstance(other_obs.state, Mapping) else float(other_obs.state.x)
            oy = float(other_obs.state["y"]) if isinstance(other_obs.state, Mapping) else float(other_obs.state.y)
        except (KeyError, TypeError, AttributeError, ValueError):
            return False

        dist = math.hypot(sx - ox, sy - oy)
        last_dist = self._last_dist_to_yield.get(other_name)
        if last_dist is None:
            self._last_dist_to_yield[other_name] = dist
            self._yield_recede_count[other_name] = 0
            self._yield_recede_latched.pop(other_name, None)
            return False

        count = self._yield_recede_count.get(other_name, 0)
        if (dist - last_dist) >= dist_eps:
            count += 1
        else:
            count = 0

        self._yield_recede_count[other_name] = count
        self._last_dist_to_yield[other_name] = dist

        if count >= n_confirm:
            self._yield_recede_latched[other_name] = True
        return self._yield_recede_latched.get(other_name, False)

    def _find_couloir_entry_point_ahead(
        self,
        state: object,
        lookahead: float = 4.0,
        step: float = 0.2,
    ) -> tuple[float, float, tuple[float, float]] | None:
        path = self.current_path
        if not path or len(path) < 2:
            return None

        try:
            sx = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
            sy = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
        except (KeyError, TypeError, AttributeError, ValueError):
            return None

        lookahead = float(lookahead)
        if lookahead <= 0.0:
            return None

        step = float(step) if step > 1e-6 else 0.1

        # Trouver l'index du point de path le plus proche du robot
        best_idx = 0
        best_dist = float("inf")
        for idx, (px_path, py_path) in enumerate(path):
            d = math.hypot(px_path - sx, py_path - sy)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        start_idx = max(0, best_idx - 1)
        start_idx = min(start_idx, len(path) - 2)

        dist_traversed = 0.0
        for seg_idx in range(start_idx, len(path) - 1):
            x0, y0 = path[seg_idx]
            x1, y1 = path[seg_idx + 1]

            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)

            if seg_len < 1e-6:
                if dist_traversed >= lookahead:
                    break
                continue

            ux = dx / seg_len
            uy = dy / seg_len

            # Dernier point "hors couloir" rencontré en avançant sur l'axe
            last_out: tuple[float, float] | None = None

            s_local = 0.0
            while s_local <= seg_len:
                if dist_traversed + s_local > lookahead:
                    break

                sample_x = x0 + ux * s_local
                sample_y = y0 + uy * s_local

                in_corridor = self._point_dans_couloir(sample_x, sample_y, (ux, uy))
                if in_corridor:
                    # On renvoie le dernier point dehors (avant-dernier)
                    if last_out is not None:
                        ex, ey = last_out
                        return (ex, ey, (ux, uy))
                    # Cas extrême: le tout premier sample est déjà "dans couloir"
                    return (sample_x, sample_y, (ux, uy))

                last_out = (sample_x, sample_y)
                s_local += step

            # Si on n'a pas échantillonné exactement l'extrémité x1,y1, on la teste
            if dist_traversed + seg_len <= lookahead:
                if self._point_dans_couloir(x1, y1, (ux, uy)):
                    if last_out is not None:
                        ex, ey = last_out
                        return (ex, ey, (ux, uy))
                    return (x1, y1, (ux, uy))

            dist_traversed += seg_len
            if dist_traversed >= lookahead:
                break

        return None

    def _project_point_to_path_arclength(
        self,
        path: list[tuple[float, float]],
        px: float,
        py: float,
    ) -> float | None:
        if not path or len(path) < 2:
            return None
        best_d2 = float("inf")
        best_s = 0.0
        s_acc = 0.0
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            vx = x1 - x0
            vy = y1 - y0
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-12:
                continue
            seg_len = math.sqrt(seg_len2)
            wx = px - x0
            wy = py - y0
            t = (wx * vx + wy * vy) / seg_len2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj_x = x0 + t * vx
            proj_y = y0 + t * vy
            dx = px - proj_x
            dy = py - proj_y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_s = s_acc + t * seg_len
            s_acc += seg_len
        return float(best_s)

    def _point_at_path_arclength(
        self,
        path: list[tuple[float, float]],
        s_query: float,
    ) -> tuple[float, float] | None:
        if not path or len(path) < 2:
            return None
        s = float(max(0.0, s_query))
        s_acc = 0.0
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            seg_len = float(math.hypot(x1 - x0, y1 - y0))
            if seg_len < 1e-12:
                continue
            if s <= s_acc + seg_len:
                t = (s - s_acc) / seg_len
                return (float(x0 + t * (x1 - x0)), float(y0 + t * (y1 - y0)))
            s_acc += seg_len
        return (float(path[-1][0]), float(path[-1][1]))

    def _diffdrive_goto_point(
        self,
        state: object,
        goal_x: float,
        goal_y: float,
        v_cap: float = 1,
        reverse_if_behind: bool = True,
        stop_dist: float = 0.03,
    ) -> DiffDriveCommands:
        """Commande diff-drive simple pour rejoindre un point (goal_x, goal_y).

        - Loi P sur distance (k_v) + loi P sur erreur d'angle (k_w).
        - Si reverse_if_behind=True et si le point est derrière, on autorise la marche arrière
          pour reculer proprement sans faire demi-tour.
        """
        sx = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
        sy = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
        psi = float(state["psi"]) if isinstance(state, Mapping) else float(state.psi)

        dx = float(goal_x) - sx
        dy = float(goal_y) - sy
        dist = float(math.hypot(dx, dy))
        if dist <= float(stop_dist):
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        target_angle = float(np.arctan2(dy, dx))
        angle_error = self._wrap_to_pi(target_angle - psi)

        # Si le point est derrière (|err| > 90°), on préfère reculer droit plutôt que tourner sur place.
        v_sign = 1.0
        if reverse_if_behind and abs(angle_error) > (np.pi / 2):
            v_sign = -1.0
            angle_error = self._wrap_to_pi((target_angle + np.pi) - psi)

        omega_min, omega_max = self.sp.omega_limits
        Rw = float(self.sg.wheelradius)
        Lw = float(self.sg.wheelbase)
        v_max = Rw * float(omega_max)

        k_v = 1.2
        k_w = float(getattr(self, "k_w", 2.0))

        v = v_sign * min(k_v * dist, float(v_cap), v_max)
        omega_body = k_w * float(angle_error)

        omega_l = v / Rw - (omega_body * Lw) / (2.0 * Rw)
        omega_r = v / Rw + (omega_body * Lw) / (2.0 * Rw)

        omega_l = float(np.clip(omega_l, omega_min, omega_max))
        omega_r = float(np.clip(omega_r, omega_min, omega_max))
        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

    def _dbg_np_file(self, sim_obs: SimObservations, msg: str) -> None:
        """Append a lightweight debug line for non-priority logic."""
        if not getattr(self, "DEBUG_NONPRIO", True):
            return
        t = float(getattr(sim_obs, "time", 0.0))
        path = getattr(self, "dbg_np_path", "nonprio_debug.log")
        line = f"[t={t:8.3f}] [NONPRIO] {msg}\n"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    @staticmethod
    def _get_player_obs(sim_obs: SimObservations, name):
        try:
            return sim_obs.players.get(name)
        except Exception:
            return None

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:

        # plus de tâches -> stop
        if not self.plan or self.current_task_idx >= len(self.plan):
            cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
            return cmd

        state = sim_obs.players[self.name].state
        sx = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
        sy = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
        psi = float(state["psi"]) if isinstance(state, Mapping) else float(state.psi)
        robot_pos = (sx, sy)
        other_players = {name: obs for name, obs in sim_obs.players.items() if name != self.name}
        sim_time = float(getattr(sim_obs, "time", 0.0))
        R = float(self.sg.radius)
        if sim_time >= self.couloir_ignore_until:
            self.couloir_ignore_other = None
            self.couloir_ignore_until = 0.0

        if self.detour_active:
            self._try_early_rejoin(robot_pos, sim_time, other_players)

        self._update_continuous_target(sim_obs)

        self.dans_couloir = self._detecter_couloir_local(state)
        self.couloir_devant = self._va_entrer_dans_couloir(state)

        if self.couloir_backoff_active and (self.couloir_backoff_target is not None):
            hx, hy = self.couloir_backoff_target
            dist_to_hold = math.hypot(sx - hx, sy - hy)
            released = False
            thr = -2.0 * R
            release_on_recede = False
            receding = False
            if self.couloir_yield_to:
                receding = self._is_yield_other_receding(sim_obs, self.couloir_yield_to)
                other_obs = self._get_player_obs(sim_obs, self.couloir_yield_to)
                if receding and other_obs is not None:
                    ost = other_obs.state
                    ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
                    oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)
                    dist_centers = math.hypot(sx - ox, sy - oy)
                    if dist_centers > 3.0 * R:
                        release_on_recede = True

            if release_on_recede:
                self._last_dist_to_yield.pop(self.couloir_yield_to, None)
                self._yield_recede_count.pop(self.couloir_yield_to, None)
                self._yield_recede_latched.pop(self.couloir_yield_to, None)
                self.couloir_ignore_until = sim_time + 5.0
                self.couloir_ignore_other = self.couloir_yield_to
                self.couloir_backoff_active = False
                self.couloir_backoff_target = None
                self.couloir_wait_exit = False
                self.couloir_yield_to = None
                self.couloir_entry_latched = None
                released = True
            elif self.couloir_yield_to and self.couloir_entry_latched:
                ex, ey, ux, uy = self.couloir_entry_latched
                other_obs = self._get_player_obs(sim_obs, self.couloir_yield_to)
                if other_obs is None:
                    self._dbg_np_file(sim_obs, "CHECK_RELEASE@BACKOFF other=None => keep backing")
                else:
                    ost = other_obs.state
                    ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
                    oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)
                    s_other = (ox - ex) * ux + (oy - ey) * uy
                    if s_other <= thr:
                        self._last_dist_to_yield.pop(self.couloir_yield_to, None)
                        self._yield_recede_count.pop(self.couloir_yield_to, None)
                        self._yield_recede_latched.pop(self.couloir_yield_to, None)
                        self.couloir_ignore_until = sim_time + 5.0
                        self.couloir_ignore_other = self.couloir_yield_to
                        self.couloir_backoff_active = False
                        self.couloir_backoff_target = None
                        self.couloir_wait_exit = False
                        self.couloir_yield_to = None
                        self.couloir_entry_latched = None
                        released = True
            if released or (not self.couloir_backoff_active) or (self.couloir_backoff_target is None):
                pass
            else:
                if dist_to_hold > 0.05:
                    cmd = self._diffdrive_goto_point(state, hx, hy, v_cap=0.25, reverse_if_behind=True)
                    return cmd
                self.couloir_backoff_active = False
                self.couloir_backoff_target = None
                self.couloir_wait_exit = True
                cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
                return cmd

        if self.couloir_wait_exit and self.couloir_yield_to and self.couloir_entry_latched:
            ex, ey, ux, uy = self.couloir_entry_latched
            thr = -2.0 * R

            other_obs = self._get_player_obs(sim_obs, self.couloir_yield_to)
            receding = self._is_yield_other_receding(sim_obs, self.couloir_yield_to)
            release_on_recede = False
            if receding and other_obs is not None:
                ost = other_obs.state
                ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
                oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)
                dist_centers = math.hypot(sx - ox, sy - oy)
                if dist_centers > 3.0 * R:
                    release_on_recede = True
            if release_on_recede:
                self._last_dist_to_yield.pop(self.couloir_yield_to, None)
                self._yield_recede_count.pop(self.couloir_yield_to, None)
                self._yield_recede_latched.pop(self.couloir_yield_to, None)
                self.couloir_ignore_until = sim_time + 5.0
                self.couloir_ignore_other = self.couloir_yield_to
                self.couloir_wait_exit = False
                self.couloir_yield_to = None
                self.couloir_entry_latched = None
            else:
                if other_obs is None:
                    cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
                    return cmd
                ost = other_obs.state
                ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
                oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)
                s_other = (ox - ex) * ux + (oy - ey) * uy

                if s_other <= thr:
                    self._last_dist_to_yield.pop(self.couloir_yield_to, None)
                    self._yield_recede_count.pop(self.couloir_yield_to, None)
                    self._yield_recede_latched.pop(self.couloir_yield_to, None)
                    self.couloir_ignore_until = sim_time + 5.0
                    self.couloir_ignore_other = self.couloir_yield_to
                    self.couloir_wait_exit = False
                    self.couloir_yield_to = None
                    self.couloir_entry_latched = None
                else:
                    cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
                    return cmd

        entry_release = self._find_couloir_entry_point_ahead(state)
        if entry_release is not None:
            ex, ey, u_hat_entry = entry_release
            ux, uy = u_hat_entry
            vx, vy = (-uy, ux)
        else:
            ex = ey = None
            ux = uy = vx = 0.0

        to_release: list[str] = []
        me_in_zone = bool(self.dans_couloir or self.couloir_devant)
        path_tube = 2.0 * R + 0.20
        for other_name in list(self.corridor_lock.keys()):
            other_obs = other_players.get(other_name)
            if other_obs is None:
                continue
            ost = other_obs.state
            ox = float(ost["x"]) if isinstance(ost, Mapping) else float(ost.x)
            oy = float(ost["y"]) if isinstance(ost, Mapping) else float(ost.y)

            dist = math.hypot(ox - sx, oy - sy)

            other_in_corr = False
            try:
                other_in_corr = bool(self._detecter_couloir_local(ost))
            except Exception:
                other_in_corr = False

            other_in_entry = False
            if ex is not None:
                dx = ox - ex
                dy = oy - ey
                s = dx * ux + dy * uy
                d = dx * vx + dy * vy
                L_entree = 2.0 * R
                w_entree = 2.0 * R + 0.2
                r_entree = 3.0 * R
                other_in_entry = (0.0 <= s <= L_entree) and (abs(d) <= w_entree) and (math.hypot(dx, dy) <= r_entree)

            other_in_zone = bool(other_in_corr or other_in_entry)
            other_on_path = self._is_other_on_my_path((ox, oy), radius=path_tube, max_segments=12)

            if (not me_in_zone) and (not other_in_zone) and (dist > 3.0 * R) and (not other_on_path):
                to_release.append(other_name)

        for other_name in to_release:
            self.corridor_lock.pop(other_name, None)
            self.corridor_lock_active.discard(other_name)
        if self.couloir_devant and not self.dans_couloir:
            target = self.current_target
            L_entree = 2.0 * R
            w_entree = 2.0 * R + 0.2
            r_entree = 3.0 * R
            u_hat: tuple[float, float] | None = None
            v_hat: tuple[float, float] | None = None
            if target is not None:
                tx = target[0] - sx
                ty = target[1] - sy
                dist_t = math.hypot(tx, ty)
                if dist_t >= 1e-6:
                    u_hat = (tx / dist_t, ty / dist_t)
                    v_hat = (-u_hat[1], u_hat[0])

            conflit_couloir = False
            conflict_other: str | None = None
            path_tube = 2.0 * R + 0.20
            for other_name, other_obs in other_players.items():
                if (other_name == self.couloir_ignore_other) and (sim_time < self.couloir_ignore_until):
                    continue
                other_state = other_obs.state
                try:
                    if self._detecter_couloir_local(other_state):
                        conflit_couloir = True
                        conflict_other = other_name
                        break
                except Exception:
                    pass

                if u_hat is None or v_hat is None:
                    continue

                ox = float(other_state["x"]) if isinstance(other_state, Mapping) else float(other_state.x)
                oy = float(other_state["y"]) if isinstance(other_state, Mapping) else float(other_state.y)
                if self._is_other_on_my_path((ox, oy), radius=path_tube, max_segments=12):
                    conflit_couloir = True
                    conflict_other = other_name
                    break
                diff_x = ox - sx
                diff_y = oy - sy
                s = diff_x * u_hat[0] + diff_y * u_hat[1]
                d = diff_x * v_hat[0] + diff_y * v_hat[1]
                dist = math.hypot(diff_x, diff_y)
                if (0.0 <= s <= L_entree) and (abs(d) <= w_entree) and (dist <= r_entree):
                    conflit_couloir = True
                    conflict_other = other_name
                    break
            if conflit_couloir:
                entry = self._find_couloir_entry_point_ahead(state)
                if entry is None:
                    cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
                    return cmd
                ex, ey, u_hat_entry = entry

                other_priority = False
                if conflict_other is not None:
                    if conflict_other not in self.corridor_lock:
                        self.corridor_lock[conflict_other] = self._am_i_priority_vs(conflict_other)
                    self.corridor_lock_active.add(conflict_other)
                    other_priority = self.corridor_lock.get(conflict_other, False)

                if not other_priority:
                    if conflict_other is not None and self.couloir_yield_to is None:
                        self.couloir_yield_to = conflict_other
                        self.couloir_entry_latched = (
                            float(ex),
                            float(ey),
                            float(u_hat_entry[0]),
                            float(u_hat_entry[1]),
                        )

                    path = self.current_path
                    if not self.couloir_backoff_active:
                        s_entry = self._project_point_to_path_arclength(path, ex, ey)
                        if s_entry is None:
                            cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

                            return cmd

                        s_hold = s_entry - (2.5 * R)
                        hold_pt = self._point_at_path_arclength(path, s_hold)
                        if hold_pt is None:
                            cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

                            return cmd

                        self.couloir_backoff_active = True
                        self.couloir_backoff_target = hold_pt

                    hx, hy = self.couloir_backoff_target or (sx, sy)
                    self.couloir_backoff_active = True
                    self.couloir_backoff_target = (hx, hy)
                    self.couloir_wait_exit = False
                    cmd = self._diffdrive_goto_point(state, hx, hy, v_cap=1, reverse_if_behind=True)

                    return cmd

        if self.yielding and self.yield_cp_id is not None and self.yield_to_agent is not None:
            cp_xy = self.cp_positions.get(self.yield_cp_id)
            if cp_xy is None:
                self.yielding = False
                self.yield_cp_id = None
                self.yield_to_agent = None
            else:
                exit_radius = self.yield_exit_radius_mult * R
                other_obs = sim_obs.players.get(self.yield_to_agent)
                still_conflict = False
                if other_obs is not None:
                    ox = (
                        float(other_obs.state["x"])
                        if isinstance(other_obs.state, Mapping)
                        else float(other_obs.state.x)
                    )
                    oy = (
                        float(other_obs.state["y"])
                        if isinstance(other_obs.state, Mapping)
                        else float(other_obs.state.y)
                    )
                    target_xy = self.current_target
                    if target_xy is not None:
                        target_dx = target_xy[0] - sx
                        target_dy = target_xy[1] - sy
                        target_dist = float(np.hypot(target_dx, target_dy))
                        if target_dist >= 1e-6:
                            u_hat = (target_dx / target_dist, target_dy / target_dist)
                            v_hat = (-u_hat[1], u_hat[0])
                            diff_cx = ox - sx
                            diff_cy = oy - sy
                            s = diff_cx * u_hat[0] + diff_cy * u_hat[1]
                            d = diff_cx * v_hat[0] + diff_cy * v_hat[1]
                            L = 6.0 * R
                            w = 2.0 * R + 0.2
                            still_conflict = (0.0 <= s <= L) and (abs(d) <= w)
                if still_conflict:
                    still_conflict = self._is_in_cp_radius(
                        sim_obs,
                        self.yield_to_agent,
                        cp_xy,
                        exit_radius,
                    )
                if still_conflict:
                    cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

                    return cmd
                self.yielding = False
                self.yield_cp_id = None
                self.yield_to_agent = None

        cp_conflict_detected = False
        i_am_priority_in_cp_conflict = True

        self._prune_cp_priority_locks(sim_obs)

        if other_players:
            cp_stop_radius = 6.0 * R
            for cp_id, (cx, cy) in self.cp_positions.items():
                dist_me = float(np.hypot(state.x - cx, state.y - cy))
                if dist_me > cp_stop_radius:
                    continue
                cp_xy = (cx, cy)
                cp_id_key = str(cp_id)

                for other_name, other_obs in other_players.items():
                    other_state = other_obs.state
                    ox = other_state["x"] if isinstance(other_state, Mapping) else other_state.x
                    oy = other_state["y"] if isinstance(other_state, Mapping) else other_state.y

                    dist_other = float(np.hypot(ox - cx, oy - cy))
                    if dist_other > cp_stop_radius:
                        continue

                    lock_key = (str(other_name), cp_id_key)
                    lock = self.cp_priority_lock.get(lock_key)
                    if lock is None:
                        lock = self._has_cp_priority_over(
                            sim_obs,
                            other_name,
                            cp_id_key,
                            dist_me,
                            dist_other,
                        )
                        self.cp_priority_lock[lock_key] = lock

                    if lock:
                        cp_conflict_detected = True
                        i_am_priority_in_cp_conflict = True
                        continue

                    my_task_cp_id = (
                        self.plan[self.current_task_idx].cp_id
                        if self.plan and self.current_task_idx < len(self.plan)
                        else None
                    )
                    my_intends_this_cp = (self.phase == "cp") and (my_task_cp_id == cp_id_key)
                    my_converge_cp = self._is_converging_to_point(sim_obs, cp_xy, key=cp_id_key)
                    my_converge_other = self._is_converging_to_agent(sim_obs, other_name)
                    my_in_cp_radius = dist_me <= cp_stop_radius

                    if not (my_in_cp_radius or my_intends_this_cp or my_converge_cp or my_converge_other):
                        continue

                    cp_conflict_detected = True
                    i_am_priority_in_cp_conflict = False
                    self.yielding = True
                    self.yield_cp_id = cp_id_key
                    self.yield_to_agent = other_name
                    cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)
                    return cmd

        should_apply_dynamic = (cp_conflict_detected) and i_am_priority_in_cp_conflict
        if should_apply_dynamic and not self.detour_active:
            self._apply_dynamic_detour_if_needed(robot_pos, sim_time, other_players)
        elif self.detour_active:
            print(f"SKIP DYNAMIC DETOUR (detour_active) @ {sim_time:.2f}")

        # path tracking : mise à jour de la cible le long du chemin global
        self._update_continuous_target(sim_obs)

        if self.current_target is None:
            cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

            return cmd

        min_clearance = float("inf")
        R = float(self.sg.radius)

        for other_name, other_obs in other_players.items():
            other_state = other_obs.state
            ox = other_state["x"] if isinstance(other_state, Mapping) else other_state.x
            oy = other_state["y"] if isinstance(other_state, Mapping) else other_state.y

            dx_c = state.x - ox
            dy_c = state.y - oy

            dist = float(np.hypot(dx_c, dy_c))  # distance euclidienne centre-à-centre
            clearance = dist - 2 * R  # distance entre coques

            if clearance < min_clearance:
                min_clearance = clearance

        target_x, target_y = self.current_target
        dx = target_x - state.x
        dy = target_y - state.y
        distance_to_target = float(np.hypot(dx, dy))

        if self.current_path:
            last_x, last_y = self.current_path[-1]
            dist_to_end = float(np.hypot(last_x - state.x, last_y - state.y))
        else:
            dist_to_end = distance_to_target

        phase_transitioned = False
        if dist_to_end < self.goal_tol:
            phase_changed = False
            current_task = self.plan[self.current_task_idx]
            if self.phase == "goal":
                self.phase = "cp"
                phase_changed = True
            elif self.phase == "cp":
                if current_task.exit_waypoints:
                    self.phase = "exit"
                    phase_changed = True
                else:
                    self.current_task_idx += 1
                    self.phase = "goal"
                    phase_changed = True
            elif self.phase == "exit":
                self.current_task_idx += 1
                self.phase = "goal"
                phase_changed = True

            if phase_changed:
                self.current_path = []
                self.current_target = None
                self.path_progress_idx = 0
                phase_transitioned = True

        if phase_transitioned:
            self._update_continuous_target(sim_obs)
            if self.current_target is None:
                cmd = DiffDriveCommands(omega_l=0.0, omega_r=0.0)

                return cmd
            target_x, target_y = self.current_target
            dx = target_x - state.x
            dy = target_y - state.y
            distance_to_target = float(np.hypot(dx, dy))

        # Le reste de get_commands (évitement d'obstacles + calcul v/omega) reste identique :
        # - on garde la logique de potentiel répulsif,
        # - on garde l'utilisation de angle_to_target, angle_error, etc.
        # Simplement, angle_to_target doit maintenant être calculé à partir de dx/dy :
        angle_to_target = float(np.arctan2(dy, dx))

        # (ne pas réintroduire path / current_wp_idx dans cette méthode)

        # Obstacle avoidance (repulsive potential)
        robot_point = Point(state.x, state.y)
        rep_vec = np.array([0.0, 0.0], dtype=float)

        # --- Champ répulsif dynamique dû aux autres agents ---
        # distance minimale aux autres agents (pour moduler la vitesse plus tard)
        min_dyn_dist = np.inf

        # rayon d'action du champ répulsif dynamique (1 m)
        dynamic_avoid_radius = 1.0

        for other_name, other_obs in other_players.items():
            other_state = other_obs.state
            dx_o = state.x - other_state.x
            dy_o = state.y - other_state.y
            dist = float(np.hypot(dx_o, dy_o))

            if dist < min_dyn_dist:
                min_dyn_dist = dist

            # champ répulsif uniquement si dist <= 1 m
            if 0.0 < dist < dynamic_avoid_radius:
                rep_vec += np.array([dx_o, dy_o], dtype=float) / (dist**2)

        # --- Champ répulsif des obstacles statiques (inchangé) ---
        for obs in self.static_obstacles:
            poly = obs.shape
            dist = robot_point.distance(poly)
            if dist < self.avoid_radius and dist > 1e-6:
                nearest_on_poly = nearest_points(robot_point, poly)[1]
                vec = np.array([state.x - nearest_on_poly.x, state.y - nearest_on_poly.y], dtype=float)
                rep_vec += vec / (dist**2)

        if np.linalg.norm(rep_vec) > 1e-8:
            avoid_angle = float(np.arctan2(rep_vec[1], rep_vec[0]))
            a1 = angle_to_target
            a2 = avoid_angle
            blended = np.arctan2(
                np.sin(a1) + self.k_avoid * np.sin(a2),
                np.cos(a1) + self.k_avoid * np.cos(a2),
            )
            angle_to_target = float(blended)

        reverse_mode = self.reverse_mode
        effective_heading = state.psi + (np.pi if reverse_mode else 0.0)
        angle_error = self._wrap_to_pi(angle_to_target - effective_heading)

        omega_min, omega_max = self.sp.omega_limits
        v_max = self.sg.wheelradius * omega_max

        v_nominal = v_max
        stop_causes: List[str] = []

        angle = abs(angle_error)
        angle_slow = np.deg2rad(20.0)
        angle_stop = np.deg2rad(120.0) if reverse_mode else np.deg2rad(80.0)

        if angle <= angle_slow:
            # petit angle -> vitesse nominale
            v = v_nominal
        elif angle >= angle_stop:
            # trop tourné -> on arrête la translation, on laisse la rotation corriger
            v = 0.0
            stop_causes.append(f"turn_in_place({angle_stop:.2f})")
        else:
            # interpolation linéaire entre v_nominal et 0
            alpha = (angle - angle_slow) / (angle_stop - angle_slow)
            v = (1.0 - alpha) * v_nominal

        if self._blocked_by_dynamic:
            v = 0.0
            stop_causes.append("blocked_by_dynamic")

        # Réduction directe de la vitesse en fonction de la clearance (euclidienne)
        if np.isfinite(min_clearance):
            if min_clearance <= 0.2:
                factor = 0.0
                stop_causes.append("clearance=0")
            elif min_clearance < 0.5:
                factor = 0.5 + 0.5 * ((min_clearance - 0.2) / (0.3))
                if factor == 0.0:
                    stop_causes.append("clearance=0")
            else:
                factor = 1.0
            v = v * factor

        # Modulation de la vitesse en fonction de la distance aux autres agents
        if np.isfinite(min_dyn_dist):
            if min_dyn_dist <= 0.2:
                v = 0.0
                stop_causes.append("dynamic_too_close")
            elif min_dyn_dist < 0.5:
                beta = (min_dyn_dist - 0.2) / (0.5 - 0.2)
                beta = float(np.clip(beta, 0.0, 1.0))
                v = v * beta

        v = -abs(v) if reverse_mode else abs(v)

        if getattr(self, "DEBUG_REVERSE_MODE", False):
            now = sim_time
            if now - self._last_reverse_log_time >= 1.0:
                print(
                    f"[REV] t={sim_time:.2f} idx={self.path_progress_idx} "
                    f"reverse={reverse_mode} angle_err={angle_error:.3f} v={v:.3f}"
                )
                self._last_reverse_log_time = now

        # Vitesse angulaire toujours proportionnelle à l'angle d'erreur
        omega_body = self.k_w * angle_error

        # On peut garder un clip sur v par sécurité, même si v <= v_nominal <= v_max
        v = float(np.clip(v, -v_max, v_max))

        # Conversion (v, omega_body) -> vitesses de roues
        R = self.sg.wheelradius
        L = self.sg.wheelbase
        omega_l = v / R - (omega_body * L) / (2 * R)
        omega_r = v / R + (omega_body * L) / (2 * R)

        omega_l = float(np.clip(omega_l, omega_min, omega_max))
        omega_r = float(np.clip(omega_r, omega_min, omega_max))

        cmd = DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

        return cmd

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        return float(wrapped)

    def _timef(self, t) -> float:
        if t is None:
            return float(self.last_replan_time) + 0.1
        try:
            return float(t)
        except Exception:
            return float(self.last_replan_time)

    def _start_detour(self, sim_time: float, end_idx: int) -> None:
        self.detour_active = True
        self.detour_end_idx = end_idx
        self.detour_start_time = float(sim_time)
        self.last_detour_time = float(sim_time)
        self.early_rejoin_hits = 0
        self.early_rejoin_last_min_dist = None
        print(f"DETOUR START @ {sim_time:.2f} (end_idx={end_idx})")

    def _end_detour(self, sim_time: float, reason: str | None = None) -> None:
        if self.detour_active:
            label = f" ({reason})" if reason else ""
            print(f"DETOUR END{label} @ {sim_time:.2f}")
        self.detour_active = False
        self.detour_end_idx = None
        self.detour_start_time = None
        self.last_detour_time = float(sim_time)

    def _maybe_end_detour(self, sim_time: float, px: float, py: float) -> None:
        if not self.detour_active or self.detour_end_idx is None:
            return
        if self.detour_end_idx >= len(self.current_path):
            return

        if self.path_progress_idx >= self.detour_end_idx:
            self._end_detour(sim_time, reason="passed_end")
            return

        end_pt = self.current_path[self.detour_end_idx]
        dist_end = float(np.hypot(px - end_pt[0], py - end_pt[1]))
        if dist_end < self.lookahead_dist * 0.5:
            self._end_detour(sim_time, reason="close_end")

    def _try_early_rejoin(
        self,
        robot_pos: tuple[float, float],
        sim_time: float,
        other_players: Mapping[PlayerName, SimObservations],
    ) -> bool:
        sim_time = float(sim_time)
        if not self.detour_active or self.detour_end_idx is None:
            self.early_rejoin_hits = 0
            return False
        path = self.current_path
        if not path:
            self.early_rejoin_hits = 0
            return False
        start_j = self.detour_end_idx
        if start_j >= len(path):
            self.early_rejoin_hits = 0
            return False

        Sx = float(robot_pos[0])
        Sy = float(robot_pos[1])
        S = (Sx, Sy)
        R = float(self.sg.radius)
        avoid_r = 2.0 * R + 0.2
        horizon_len = 4.0 * R
        dist_margin = avoid_r + 0.1

        other_positions: list[tuple[float, float]] = []
        min_dist_to_others = float("inf")
        for other_obs in other_players.values():
            state = other_obs.state
            ox = float(state["x"]) if isinstance(state, Mapping) else float(state.x)
            oy = float(state["y"]) if isinstance(state, Mapping) else float(state.y)
            other_positions.append((ox, oy))
            dist = math.hypot(Sx - ox, Sy - oy)
            if dist < min_dist_to_others:
                min_dist_to_others = dist

        buffer_margin = R + 0.05
        static_buffers: list = []
        for obs in self.static_obstacles:
            try:
                static_buffers.append(obs.shape.buffer(buffer_margin))
            except Exception:
                continue

        candidate_idx: int | None = None
        candidate_pt: tuple[float, float] | None = None
        acc = 0.0
        for j in range(start_j, len(path)):
            if j > start_j:
                prev = path[j - 1]
                curr = path[j]
                acc += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
                if acc > horizon_len:
                    break
            candidate = path[j]
            segment_free = True
            for other_pos in other_positions:
                if self._segment_intersects_disk(S, candidate, other_pos, avoid_r):
                    segment_free = False
                    break
            if not segment_free:
                continue
            line = LineString([S, candidate])
            for buf in static_buffers:
                if line.intersects(buf):
                    segment_free = False
                    break
            if not segment_free:
                continue
            candidate_idx = j
            candidate_pt = candidate
            break

        last_min = self.early_rejoin_last_min_dist
        safe_distance = min_dist_to_others > dist_margin
        distance_increasing = last_min is None or (min_dist_to_others >= last_min - 1e-6)
        cooldown_ok = sim_time >= self.early_rejoin_cooldown_until
        self.early_rejoin_last_min_dist = min_dist_to_others

        rejoin_ok = candidate_idx is not None and safe_distance and distance_increasing and cooldown_ok
        if not rejoin_ok:
            self.early_rejoin_hits = 0
            return False

        self.early_rejoin_hits += 1
        if self.early_rejoin_hits < self.early_rejoin_required_hits:
            return False

        idx = candidate_idx
        pt = candidate_pt or path[idx]
        new_path = [S, pt] + path[idx + 1 :]
        self.current_path = new_path
        self.path_progress_idx = 0
        self.last_global_rebuild_time = sim_time
        self._end_detour(sim_time, reason="early_rejoin")
        self.early_rejoin_cooldown_until = sim_time + 0.6
        self.early_rejoin_hits = 0
        return True

    def _is_in_cp_radius(
        self,
        sim_obs: SimObservations,
        agent_name: PlayerName,
        cp_xy: tuple[float, float],
        radius: float,
    ) -> bool:
        obs = sim_obs.players.get(agent_name)
        if obs is None:
            return False
        st = obs.state
        x = float(st["x"]) if isinstance(st, Mapping) else float(st.x)
        y = float(st["y"]) if isinstance(st, Mapping) else float(st.y)
        return float(np.hypot(x - cp_xy[0], y - cp_xy[1])) <= radius


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    Industrial-grade global planner performing MT-MRTA with PD-VRP style ordering,
    congestion-aware costs, and obstacle-aware shortest-path distances.
    """

    ALPHA = 2.0  # density reward weight
    BETA = 1.0  # isolation penalty weight
    GAMMA = 0.5  # travel-time weight
    LAMBDA = 1.5  # congestion penalty weight
    DENSITY_RADIUS = 5.0
    LARGE_COST = 1e6

    def __init__(self):
        self.agent_geometry: Dict[PlayerName, Dict[str, float]] = {}
        self.agent_dynamics: Dict[PlayerName, Dict[str, float]] = {}
        self.agent_initial_positions: Dict[PlayerName, Tuple[float, float]] = {}
        self.collection_points: Dict[str, Tuple[float, float]] = {}
        self.goals_positions: Dict[str, Tuple[float, float]] = {}
        self.static_obstacles: List[StaticObstacle] = []
        self.node_coords: List[Tuple[float, float]] = []
        self.node_labels: List[Tuple[str, int | str]] = []
        self.agent_node_idx: Dict[PlayerName, int] = {}
        self.goal_node_idx: Dict[str, int] = {}
        self.cp_node_idx: Dict[str, int] = {}
        self.shortest_paths: np.ndarray | None = None
        self.goal_density: Dict[str, float] = {}
        self.goal_isolation: Dict[str, float] = {}
        self.goal_best_cp: Dict[str, str] = {}
        self.graph_csr: csr_matrix | None = None
        self.predecessors: np.ndarray | None = None
        self.safety_margin: float = 0.1
        self.inflated_obstacles: List = []
        self.inflation_radius: float = 0.0
        # --- nouveaux attributs pour les "corner caps" sur les sommets ---
        self.corner_caps: List = []  # liste de géométries Shapely (polygons ou multipolygons)
        self.CORNER_CAP_RADIUS: float = 0.4  # rayon local des caps autour des sommets
        self.CORNER_ANGLE_THRESHOLD: float = np.deg2rad(110.0)  # on ne traite que les angles aigus
        self._visibility_cache: Dict[Tuple[Tuple[float, float], Tuple[float, float]], bool] = {}
        self._static_edges: List[Tuple[int, int, float]] = []
        self._static_nodes_count: int = 0
        self._graph_static_built: bool = False
        self.WAYPOINT_MIN_DIST: float = 0.2
        
        # CP contextuel et caches de coût
        self.agent_goal_cp: Dict[PlayerName, Dict[str, str]] = {}
        self._cp_choice_cache: Dict[Tuple[PlayerName, str], str] = {}
        self._cp_travel_time_cache: Dict[Tuple[PlayerName, str], float] = {}
        # distances statiques précomputées
        self.goal_cp_dist: Dict[str, Dict[str, float]] = {}
        self.cp_goal_dist: Dict[str, Dict[str, float]] = {}
        self.agent_init_goal_dist: Dict[PlayerName, Dict[str, float]] = {}
        self.node_idx_to_cp_id: Dict[int, str] = {}
        self.agent_initial_nodes_idx: Dict[PlayerName, int] = {}
        self.ETA_WEIGHT: float = 1.0
        self.KAPPA_LOOKAHEAD: float = 0.3
        self.obstacle_tree: STRtree | None = None  # FIX: spatial index for visibility
        self.all_obstacle_geoms: List = []  # FIX: flattened obstacles for STRtree

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        self._visibility_cache.clear()
        self._ingest_agents(init_sim_obs)
        self.agent_goal_cp = {agent_name: {} for agent_name in self.agent_geometry.keys()}
        self._ingest_collection_points(init_sim_obs)
        self._ingest_goals(init_sim_obs)
        self._ingest_obstacles(init_sim_obs)
        self._build_graph()
        self._compute_shortest_paths()
        self._precompute_static_distances()
        # --- AJUSTEMENT DYNAMIQUE DE L'ANGLE EPSILON ---
        # On compte le nombre d'agents présents dans les observations
        num_agents = len(init_sim_obs.players_obs)
        
        if num_agents <= 1:
            # En solo : on veut des trajectoires très lisses. 
            # On garde peu de waypoints (angle plus grand pour simplifier)
            self.WAYPOINT_ANGLE_EPS = np.deg2rad(10.0) 
        else:
            # En multi : on veut coller précisément au graphe de visibilité
            # pour éviter les collisions avec les murs gonflés.
            self.WAYPOINT_ANGLE_EPS = np.deg2rad(0.0)
        if not self.goal_density or not self.goal_isolation:
            self._compute_goal_heuristics()
        if not self.goal_best_cp:
            self._assign_best_cp_to_goals()
        agent_assigned_goals = self._multi_round_allocation()
        ordered_agent_tasks = self._order_agent_tasks(agent_assigned_goals)
        agent_plans = self._build_agent_plans(ordered_agent_tasks)
        self._append_final_cp_exit_waypoints(agent_plans)
        print("planner agent keys:", list(agent_plans.keys()))
        for k, v in agent_plans.items():
            print("planner tasks for", k, [t.goal_id for t in v.tasks])
        global_plan_message = GlobalPlanMessage(
            agent_plans=agent_plans,
            goal_positions=self.goals_positions,
            cp_positions=self.collection_points,
            num_agents=len(self.agent_geometry),
        )
        return global_plan_message.model_dump_json(round_trip=True)

    def _ingest_agents(self, init_sim_obs: InitSimGlobalObservations) -> None:
        self.agent_geometry.clear()
        self.agent_initial_positions.clear()
        max_radius = 0.0
        for agent_name, obs in init_sim_obs.players_obs.items():
            init_state = init_sim_obs.initial_states[agent_name]
            self.agent_initial_positions[agent_name] = (init_state.x, init_state.y)
            geometry = obs.model_geometry
            omega_min, omega_max = obs.model_params.omega_limits
            # On augmente le rayon effectif du robot de 10% pour le planning
            effective_radius = geometry.radius * 1.1
            self.agent_geometry[agent_name] = {
                "wheelbase": geometry.wheelbase,
                "wheelradius": geometry.wheelradius,
                "radius": effective_radius,
                "omega_min": omega_min,
                "omega_max": omega_max,
            }
            if effective_radius > max_radius:
                max_radius = effective_radius
        # safety_margin augmenté (par défaut 0.4) pour gonfler davantage les obstacles et élargir la zone de sécurité
        self.inflation_radius = max_radius + self.safety_margin

    def _ingest_collection_points(self, init_sim_obs: InitSimGlobalObservations) -> None:
        self.collection_points.clear()
        for cp in init_sim_obs.collection_points.values():
            centroid = cp.polygon.centroid
            self.collection_points[cp.point_id] = (centroid.x, centroid.y)

    def _ingest_goals(self, init_sim_obs: InitSimGlobalObservations) -> None:
        new_goals: Dict[str, Tuple[float, float]] = {}
        for goal_id, goal in init_sim_obs.shared_goals.items():
            centroid = goal.polygon.centroid
            new_goals[goal_id] = (centroid.x, centroid.y)
        if new_goals == self.goals_positions:
            return
        self.goals_positions = new_goals
        self.goal_density = {}
        self.goal_isolation = {}
        self.goal_best_cp = {}

    def _ingest_obstacles(self, init_sim_obs: InitSimGlobalObservations) -> None:
        self.static_obstacles = list(init_sim_obs.dg_scenario.static_obstacles)
        self.inflated_obstacles = []
        self.corner_caps = []

        for obs in init_sim_obs.dg_scenario.static_obstacles:
            orig_poly = obs.shape
            inflated_poly = orig_poly.buffer(self.inflation_radius)
            self.inflated_obstacles.append(inflated_poly)

            # Zone annulaire purement externe : P⁺ \ P
            try:
                outer_ring = inflated_poly.difference(orig_poly)
            except Exception:
                outer_ring = inflated_poly

            if not hasattr(orig_poly, "exterior") or orig_poly.exterior is None:
                continue

            coords = list(orig_poly.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]

            n = len(coords)
            if n < 3:
                continue

            for i in range(n):
                x_i, y_i = coords[i]
                x_prev, y_prev = coords[(i - 1) % n]
                x_next, y_next = coords[(i + 1) % n]

                v1 = np.array([x_prev - x_i, y_prev - y_i], dtype=float)
                v2 = np.array([x_next - x_i, y_next - y_i], dtype=float)

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                v1 /= norm1
                v2 /= norm2
                dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
                angle = float(np.arccos(dot))

                # On ne crée un cap que pour les sommets "aigus"
                if angle > self.CORNER_ANGLE_THRESHOLD:
                    continue

                local_disk = Point(x_i, y_i).buffer(self.CORNER_CAP_RADIUS)

                try:
                    cap_geom = outer_ring.intersection(local_disk)
                except Exception:
                    cap_geom = None

                if cap_geom is not None and not cap_geom.is_empty:
                    self.corner_caps.append(cap_geom)

        # build STRtree for visibility checks  # FIX:
        self.all_obstacle_geoms = list(self.inflated_obstacles) + list(self.corner_caps)  # FIX:
        self._visibility_cache = {}  # FIX:
        self.obstacle_tree = STRtree(self.all_obstacle_geoms) if self.all_obstacle_geoms else None  # FIX:

    def _build_graph(self) -> None:
        self.node_coords = []
        self.node_labels = []
        self.agent_node_idx = {}
        self.goal_node_idx = {}
        self.cp_node_idx = {}
        for name, pos in self.agent_initial_positions.items():
            idx = len(self.node_coords)
            self.node_coords.append(pos)
            self.node_labels.append(("agent", name))
            self.agent_node_idx[name] = idx
        for goal_id, pos in self.goals_positions.items():
            idx = len(self.node_coords)
            self.node_coords.append(pos)
            self.node_labels.append(("goal", goal_id))
            self.goal_node_idx[goal_id] = idx
        for cp_id, pos in self.collection_points.items():
            idx = len(self.node_coords)
            self.node_coords.append(pos)
            self.node_labels.append(("cp", cp_id))
            self.cp_node_idx[cp_id] = idx
        for inflated_poly in self.inflated_obstacles:
            if hasattr(inflated_poly, "exterior") and inflated_poly.exterior is not None:
                for x, y in inflated_poly.exterior.coords:
                    self.node_coords.append((x, y))
                    self.node_labels.append(("obs", -1))
        coords_array = np.array(self.node_coords)
        full_dist = distance_matrix(coords_array, coords_array)
        visibility = np.ones_like(full_dist, dtype=bool)
        num_nodes = len(self.node_coords)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if not self._is_visible(self.node_coords[i], self.node_coords[j]):
                    visibility[i, j] = False
                    visibility[j, i] = False
        weights = np.where(visibility, full_dist, np.inf)
        self.graph_csr = csr_matrix(weights)
        self.agent_initial_nodes_idx = dict(self.agent_node_idx)

    def _is_visible(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        key = tuple(sorted((p1, p2)))  # FIX:
        if key in self._visibility_cache:  # FIX:
            return self._visibility_cache[key]  # FIX:

        if self.obstacle_tree is None:  # FIX:
            self._visibility_cache[key] = True  # FIX:
            return True  # FIX:

        segment = LineString([p1, p2])  # FIX:
        result = True  # FIX:
        try:  # FIX:
            candidate_indices = self.obstacle_tree.query(segment)  # FIX:
        except Exception:  # FIX:
            candidate_indices = []  # FIX:

        for idx in candidate_indices:  # FIX:
            try:  # FIX:
                poly = self.all_obstacle_geoms[int(idx)]  # FIX:
            except Exception:  # FIX:
                continue  # FIX:
            if segment.crosses(poly) or segment.within(poly) or poly.contains(segment):  # FIX:
                result = False  # FIX:
                break  # FIX:

        self._visibility_cache[key] = result  # FIX:
        return result  # FIX:

    def _compute_shortest_paths(self) -> None:
        if self.graph_csr is None:
            self.shortest_paths = None
            self.predecessors = None
            return
        dist_matrix, predecessors = shortest_path(
            self.graph_csr, directed=False, unweighted=False, return_predecessors=True
        )
        self.shortest_paths = dist_matrix
        self.predecessors = predecessors

    def _compute_goal_heuristics(self) -> None:
        goal_coords = np.array(list(self.goals_positions.values()))
        goal_ids = list(self.goals_positions.keys())
        if len(goal_coords) == 0:
            self.goal_density = {}
            self.goal_isolation = {}
            return
        if len(goal_coords) == 1:
            self.goal_density = {goal_ids[0]: 0.0}
            self.goal_isolation = {goal_ids[0]: 0.0}
            return
        dmat = distance_matrix(goal_coords, goal_coords)
        tree = KDTree(goal_coords)
        densities: Dict[int, float] = {}
        isolations: Dict[int, float] = {}
        for idx, gid in enumerate(goal_ids):
            neighbors = tree.query_ball_point(goal_coords[idx], r=self.DENSITY_RADIUS)
            density_val = max(len(neighbors) - 1, 0)
            densities[gid] = float(density_val)
            sorted_dists = np.sort(dmat[idx])
            nearest = sorted_dists[1] if len(sorted_dists) > 1 else 0.0
            isolations[gid] = float(nearest)
        self.goal_density = densities
        self.goal_isolation = isolations

    def _assign_best_cp_to_goals(self) -> None:
        self.goal_best_cp = {}
        for goal_id, g_idx in self.goal_node_idx.items():
            best_cp = None
            best_dist = np.inf
            for cp_id, cp_idx in self.cp_node_idx.items():
                dist = self.shortest_paths[g_idx, cp_idx]
                if dist < best_dist:
                    best_dist = dist
                    best_cp = cp_id
            if best_cp is not None:
                self.goal_best_cp[goal_id] = best_cp

    def _precompute_static_distances(self) -> None:
        if self.shortest_paths is None:
            self.goal_cp_dist = {}
            self.cp_goal_dist = {}
            self.agent_init_goal_dist = {}
            self.node_idx_to_cp_id = {}
            return
        self.goal_cp_dist = {}
        self.cp_goal_dist = {cp_id: {} for cp_id in self.cp_node_idx.keys()}
        self.node_idx_to_cp_id = {idx: cp_id for cp_id, idx in self.cp_node_idx.items()}

        for goal_id, goal_idx in self.goal_node_idx.items():
            cp_map: Dict[str, float] = {}
            for cp_id, cp_idx in self.cp_node_idx.items():
                d = self.shortest_paths[goal_idx, cp_idx]
                if np.isfinite(d):
                    cp_map[cp_id] = d
                    self.cp_goal_dist[cp_id][goal_id] = d
            self.goal_cp_dist[goal_id] = cp_map

        self.agent_init_goal_dist = {}
        for agent_name, init_pos_idx in self.agent_initial_nodes_idx.items():
            goal_map: Dict[str, float] = {}
            for goal_id, goal_idx in self.goal_node_idx.items():
                d = self.shortest_paths[init_pos_idx, goal_idx]
                if np.isfinite(d):
                    goal_map[goal_id] = d
            self.agent_init_goal_dist[agent_name] = goal_map

    def _multi_round_allocation(self) -> Dict[PlayerName, List[str]]:
        remaining_goals = set(self.goals_positions.keys())
        agents = list(self.agent_geometry.keys())
        agent_assigned_goals: Dict[PlayerName, List[str]] = {a: [] for a in agents}
        agent_virtual_positions: Dict[PlayerName, int] = {a: self.agent_initial_nodes_idx[a] for a in agents}
        agent_virtual_time: Dict[PlayerName, float] = {a: 0.0 for a in agents}
        cp_load: Dict[str, int] = {cp_id: 0 for cp_id in self.collection_points.keys()}
        while remaining_goals:
            goals_list = list(remaining_goals)
            num_agents = len(agents)
            num_goals = len(goals_list)
            if num_goals == 0:
                break
            self._cp_choice_cache.clear()
            self._cp_travel_time_cache.clear()
            cost_matrix = np.full((num_agents, num_goals), self.LARGE_COST)
            for i, agent in enumerate(agents):
                for j, goal_id in enumerate(goals_list):
                    cost, best_cp_id, travel_time = self._compute_cost(
                        agent_name=agent,
                        goal_id=goal_id,
                        current_node_idx=agent_virtual_positions[agent],
                        cp_load=cp_load,
                        current_agent_time=agent_virtual_time[agent],
                        remaining_goals=remaining_goals,
                    )
                    cost_matrix[i, j] = cost
                    if best_cp_id:
                        self._cp_choice_cache[(agent, goal_id)] = best_cp_id
                        self._cp_travel_time_cache[(agent, goal_id)] = travel_time
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_any = False
            for r, c in zip(row_ind, col_ind):
                agent = agents[r]
                goal_id = goals_list[c]
                assigned_any = True
                agent_assigned_goals[agent].append(goal_id)
                remaining_goals.discard(goal_id)
                cp_id = self._cp_choice_cache.get((agent, goal_id))
                if cp_id is not None:
                    self.agent_goal_cp[agent][goal_id] = cp_id
                    cp_load[cp_id] += 1
                    agent_virtual_positions[agent] = self.cp_node_idx[cp_id]
                    travel_time = self._cp_travel_time_cache.get((agent, goal_id), 0.0)
                    agent_virtual_time[agent] += travel_time
            if not assigned_any:
                break
        return agent_assigned_goals

    def _compute_cost(
        self,
        agent_name: PlayerName,
        goal_id: str,
        current_node_idx: int,
        cp_load: Dict[str, int],
        current_agent_time: float,
        remaining_goals: Sequence[str] | None = None,
    ) -> Tuple[float, str, float]:
        init_idx = self.agent_initial_nodes_idx.get(agent_name)
        if init_idx is not None and current_node_idx == init_idx:
            d_agent_goal = self.agent_init_goal_dist.get(agent_name, {}).get(goal_id, np.inf)
        else:
            cp_start_id = self.node_idx_to_cp_id.get(current_node_idx)
            if cp_start_id is not None:
                d_agent_goal = self.cp_goal_dist.get(cp_start_id, {}).get(goal_id, np.inf)
            else:
                goal_idx = self.goal_node_idx[goal_id]
                d_agent_goal = self.shortest_paths[current_node_idx, goal_idx]

        if not np.isfinite(d_agent_goal):
            return self.LARGE_COST, "", 0.0

        geom = self.agent_geometry[agent_name]
        vmax = geom["wheelradius"] * max(geom["omega_max"], 1e-6)

        density = self.goal_density.get(goal_id, 0.0)
        isolation = self.goal_isolation.get(goal_id, 0.0)

        best_cp_id: str | None = None
        best_cost = self.LARGE_COST
        best_travel_time = 0.0

        for cp_id, d_goal_cp in self.goal_cp_dist.get(goal_id, {}).items():
            if not np.isfinite(d_goal_cp):
                continue

            distance_total = d_agent_goal + d_goal_cp
            if distance_total <= 0.0:
                continue

            travel_time = distance_total / vmax if vmax > 0 else self.LARGE_COST
            finish_time = current_agent_time + travel_time

            congestion_penalty = self.LAMBDA * cp_load.get(cp_id, 0)

            base_cost = (
                distance_total
                + self.GAMMA * travel_time
                - self.ALPHA * density
                + self.BETA * isolation
                + congestion_penalty
                + self.ETA_WEIGHT * finish_time
            )

            future_term = 0.0
            if remaining_goals:
                candidate_next_goals = [g for g in remaining_goals if g != goal_id]
                if candidate_next_goals:
                    d_next_list = [self.cp_goal_dist.get(cp_id, {}).get(gp, np.inf) for gp in candidate_next_goals]
                    best_next = min(d_next_list) if d_next_list else np.inf
                    if np.isfinite(best_next):
                        future_term = self.KAPPA_LOOKAHEAD * best_next

            cost_cp = base_cost + future_term

            if cost_cp < best_cost:
                best_cost = cost_cp
                best_cp_id = cp_id
                best_travel_time = travel_time

        if best_cp_id is None:
            return self.LARGE_COST, "", 0.0

        return best_cost, best_cp_id, best_travel_time

    def _reconstruct_path_nodes(self, start_idx: int, end_idx: int) -> List[int]:
        if self.predecessors is None:
            return []
        if start_idx == end_idx:
            return [start_idx]
        path = [end_idx]
        current = end_idx
        max_steps = len(self.predecessors)
        steps = 0
        while current != start_idx and steps < max_steps:
            current = int(self.predecessors[start_idx, current])
            if current == -9999:
                return []
            path.append(current)
            steps += 1
        if current != start_idx:
            return []
        path.reverse()
        return path

    def _nodes_to_waypoints(self, node_path: List[int]) -> List[Tuple[float, float]]:
        if not node_path:
            return []
        raw_points: List[Tuple[float, float]] = [self.node_coords[idx] for idx in node_path]

        filtered: List[Tuple[float, float]] = []
        last_x, last_y = raw_points[0]
        filtered.append((last_x, last_y))
        for x, y in raw_points[1:]:
            dx = x - last_x
            dy = y - last_y
            if np.hypot(dx, dy) >= self.WAYPOINT_MIN_DIST:
                filtered.append((x, y))
                last_x, last_y = x, y

        if len(filtered) <= 2:
            return [(float(x), float(y)) for x, y in filtered]

        smoothed = self._shortcut_smoothing(filtered)

        if len(smoothed) <= 2:
            return [(float(x), float(y)) for x, y in smoothed]

        simplified: List[Tuple[float, float]] = []
        simplified.append(smoothed[0])
        for i in range(1, len(smoothed) - 1):
            x_prev, y_prev = smoothed[i - 1]
            x_curr, y_curr = smoothed[i]
            x_next, y_next = smoothed[i + 1]

            v1 = np.array([x_curr - x_prev, y_curr - y_prev], dtype=float)
            v2 = np.array([x_next - x_curr, y_next - y_curr], dtype=float)

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                simplified.append((x_curr, y_curr))
                continue

            v1 /= norm1
            v2 /= norm2
            dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle = np.arccos(dot)

            if angle > self.WAYPOINT_ANGLE_EPS:
                simplified.append((x_curr, y_curr))

        simplified.append(smoothed[-1])

        return [(float(x), float(y)) for x, y in simplified]

    def _shortcut_smoothing(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 2:
            return points
        smoothed: List[Tuple[float, float]] = []
        n = len(points)
        i = 0
        while i < n:
            j = n - 1
            last_visible_index = i
            while j > i + 1:
                p_i = points[i]
                p_j = points[j]
                if self._is_visible(p_i, p_j):
                    last_visible_index = j
                    break
                j -= 1

            smoothed.append(points[i])

            if last_visible_index == i:
                i += 1
            else:
                i = last_visible_index

        if smoothed[-1] != points[-1]:
            smoothed.append(points[-1])

        return smoothed

    def _route_cost(self, agent: PlayerName, route: List[int]) -> float:
        if self.shortest_paths is None:
            return self.LARGE_COST
        current_node = self.agent_node_idx[agent]
        total = 0.0
        for goal_id in route:
            goal_node = self.goal_node_idx[goal_id]
            cp_id = self.agent_goal_cp[agent].get(goal_id)
            if cp_id is None:
                return self.LARGE_COST
            cp_node = self.cp_node_idx[cp_id]
            leg = self.shortest_paths[current_node, goal_node] + self.shortest_paths[goal_node, cp_node]
            if not np.isfinite(leg):
                return self.LARGE_COST
            total += leg
            current_node = cp_node
        return total

    def _two_opt_improvement(self, agent: PlayerName, route: List[int]) -> List[int]:
        if len(route) < 3:
            return route
        try:
            best_route = list(route)
            best_cost = self._route_cost(agent, best_route)
            if not np.isfinite(best_cost):
                return route
            improved = True
            while improved:
                improved = False
                for i in range(len(best_route) - 1):
                    for j in range(i + 1, len(best_route)):
                        new_route = best_route[:i] + list(reversed(best_route[i : j + 1])) + best_route[j + 1 :]
                        new_cost = self._route_cost(agent, new_route)
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
            return best_route
        except Exception:
            return route

    def _order_agent_tasks(self, agent_assigned_goals: Dict[PlayerName, List[str]]) -> Dict[PlayerName, List[str]]:
        ordered: Dict[PlayerName, List[str]] = {}
        for agent, goals in agent_assigned_goals.items():
            remaining = set(goals)
            current_node = self.agent_node_idx[agent]
            ordered_list: List[int] = []
            while remaining:
                best_goal = None
                best_score = np.inf
                for gid in remaining:
                    goal_idx = self.goal_node_idx[gid]
                    cp_id = self.agent_goal_cp[agent].get(gid)
                    if cp_id is None:
                        continue
                    cp_idx = self.cp_node_idx[cp_id]
                    score = self.shortest_paths[current_node, goal_idx] + self.shortest_paths[goal_idx, cp_idx]
                    if score < best_score:
                        best_score = score
                        best_goal = gid
                if best_goal is None:
                    break
                ordered_list.append(best_goal)
                remaining.remove(best_goal)
                cp_id = self.agent_goal_cp[agent].get(best_goal)
                if cp_id is None:
                    break
                current_node = self.cp_node_idx[cp_id]
            improved_route = self._two_opt_improvement(agent, ordered_list)
            ordered[agent] = improved_route if improved_route else ordered_list
        return ordered

    def _append_final_cp_exit_waypoints(self, agent_plans: Dict[str, AgentPlan]) -> None:
        cp_groups: Dict[str, List[tuple[AgentPlan, GoalTask]]] = {}
        for plan in agent_plans.values():
            if not plan.tasks:
                continue
            last_task = plan.tasks[-1]
            cp_groups.setdefault(last_task.cp_id, []).append((plan, last_task))

        angles = [2.0 * math.pi * i / 16.0 for i in range(16)]
        obstacles = list(self.inflated_obstacles)
        cp_centers = self.collection_points

        for cp_id, entries in cp_groups.items():
            cp_xy = cp_centers.get(cp_id)
            if cp_xy is None:
                continue
            other_cp_centers = {cid: xy for cid, xy in cp_centers.items() if cid != cp_id}
            assigned_points: List[tuple[float, float]] = []

            for plan, last_task in entries:
                radius = self.agent_geometry.get(plan.agent_name, {}).get("radius", 1.0)
                exit_dist = 7.0 * float(radius)
                best_point = None
                best_score = float("inf")

                for angle in angles:
                    px = cp_xy[0] + exit_dist * math.cos(angle)
                    py = cp_xy[1] + exit_dist * math.sin(angle)
                    candidate = (px, py)
                    segment = LineString([cp_xy, candidate])

                    blocked = False
                    for obs in obstacles:
                        if segment.intersects(obs) or obs.contains(Point(candidate)):
                            blocked = True
                            break
                    if blocked:
                        continue

                    score = 0.0
                    for other in assigned_points:
                        dist = math.hypot(px - other[0], py - other[1])
                        score += math.exp(-dist / (exit_dist + 1e-6))
                    for other_cp_xy in other_cp_centers.values():
                        dist_cp = math.hypot(px - other_cp_xy[0], py - other_cp_xy[1])
                        score += math.exp(-dist_cp / (exit_dist + 1e-6))

                    if score < best_score:
                        best_score = score
                        best_point = candidate

                if best_point is None:
                    best_point = (cp_xy[0] + exit_dist, cp_xy[1])

                assigned_points.append(best_point)

                if not last_task.cp_waypoints:
                    last_task.cp_waypoints = [cp_xy]
                elif last_task.cp_waypoints[-1] != cp_xy:
                    last_task.cp_waypoints.append(cp_xy)
                exit_path: List[Tuple[float, float]] = [cp_xy]
                if best_point != cp_xy:
                    exit_path.append(best_point)
                last_task.exit_waypoints = exit_path

    def _build_agent_plans(self, ordered_agent_goals: Dict[PlayerName, List[str]]) -> Dict[str, AgentPlan]:
        agent_plans: Dict[str, AgentPlan] = {}
        for agent_name, goals in ordered_agent_goals.items():
            tasks: List[GoalTask] = []
            current_node_idx = self.agent_node_idx.get(agent_name, None)
            for goal_id in goals:
                cp_id = self.agent_goal_cp[agent_name].get(goal_id)
                if cp_id is None:
                    continue
                if current_node_idx is None:
                    continue
                goal_idx = self.goal_node_idx[goal_id]
                cp_idx = self.cp_node_idx[cp_id]
                path_agent_to_goal = self._reconstruct_path_nodes(current_node_idx, goal_idx)
                goal_waypoints = self._nodes_to_waypoints(path_agent_to_goal)
                path_goal_to_cp = self._reconstruct_path_nodes(goal_idx, cp_idx)
                cp_waypoints = self._nodes_to_waypoints(path_goal_to_cp)
                if not goal_waypoints and goal_id in self.goals_positions:
                    goal_waypoints = [self.goals_positions[goal_id]]
                if not cp_waypoints and cp_id in self.collection_points:
                    cp_waypoints = [self.collection_points[cp_id]]
                tasks.append(
                    GoalTask(
                        goal_id=goal_id,
                        cp_id=cp_id,
                        goal_waypoints=goal_waypoints,
                        cp_waypoints=cp_waypoints,
                    )
                )
                current_node_idx = cp_idx
            agent_plans[str(agent_name)] = AgentPlan(agent_name=agent_name, tasks=tasks)
        return agent_plans
