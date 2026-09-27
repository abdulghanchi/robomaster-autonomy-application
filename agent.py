"""
nav agent: map -> plan -> follow, replanning as the world changes.

APPROACH (top-level overview)
  every tick:  estimate pose  ->  stitch scan into map  ->  check / replan path  ->  follow path  ->  (vx, vy)

  1. pose filter.  the sim's dynamics are exact and public (clamp speed, then clamp the change in velocity), so we replay
     our own commands to know our true velocity and predict our position.  a small complementary gain pulls that
     prediction toward the noisy measurement.  jitter drops from ~2 cm to a few mm.

  2. map.  a 0.1 m grid, each cell UNKNOWN / FREE / BLOCKED.  every scan overwrites the cells it covers ("latest
     observation wins"), in both directions: free->blocked (pop-up obstacles) and blocked->free (removed walls).
     the scan's contents are cut around the robot's TRUE cell, but its origin is reported from the NOISY pose, so a raw
     scan can land one cell off.  we recompute the origin from our filtered pose instead, and when the filtered pose sits
     within a few mm of a cell boundary we try both candidate origins and keep the one that agrees best with the map.
     without this, walls jitter by 10 cm, the 0.8 m corridor (05) closes, and "latest wins" flickers.

  3. clearance.  for every cell we track how far its centre is from the nearest blocked cell, bucketed into rings
     (0.33, 0.36, 0.40 ... 0.80 m), maintained incrementally with per-ring counters when cells change.
       - ring 0 (< 0.33 m) is hard inflation: the planner never enters it.  0.33 = robot radius 0.30 + 3 cm, which still
         leaves the exact two-cell channel in the 0.8 m corridor (those cell centres are 0.35 m from each wall).
       - outer rings add a soft cost, so paths prefer open space but can still squeeze through tight gaps.
     field edges count as walls.

  4. UNKNOWN SPACE IS PLANNED AS FREE (optimistic / "free-space assumption").  why: the goal is always reachable, and
     most of the field starts unseen, so a pessimistic planner would have no path at all (03, 06, 07 need us to plan
     into the unknown and discover the wall).  it is safe because the scan reaches 2.5 m while braking from 2 m/s
     takes 0.5 m, so anything in our way shows up long before we need to react.

  5. planner.  8-connected A* with an octile heuristic.  a diagonal move is only allowed if both orthogonal neighbours
     are passable, so the path can never slip between two blocked cells that touch at a corner.  if the start or goal is
     inside hard inflation (04, or a noisy pose sample), it is snapped to the nearest passable cell by BFS.
     the raw staircase is then shortcut ("string pulling"): from each anchor, extend to the farthest waypoint still in
     line of sight.  line of sight is an exact continuous clearance check (see _los), not just a cell walk.

  6. drivable curve + speed profile.  each corner of the smoothed path is replaced by a circular arc tangent to both
     legs.  a robot at speed v with lateral acceleration A_TURN follows radius v^2 / A_TURN, so a bigger arc = a faster
     corner.  the arc is as big as the legs and the room at the corner allow, and every arc is checked for clearance
     (radius halved until it fits).  the curve is sampled every ~5 cm; each sample gets a speed cap from its curvature,
     then a backward pass guarantees we can always brake in time (v^2 = v_next^2 + 2 a d) and a forward pass keeps
     acceleration realistic.  the goal itself needs no stop: the run ends as soon as we are inside 0.25 m.

  7. follower.  the robot is holonomic, so following = velocity along the path tangent (feedforward) plus a small pull
     back onto the path (feedback).  we deliberately do NOT chase a lookahead point (pure pursuit): that cuts inside
     every arc by ~lookahead^2 / (2 R), about 3 cm on tight corners, which caused collisions in testing.

  8. REPLAN POLICY.  we keep the current path while it is valid, and replan when:
       - newly blocked cells make the remaining path unsafe (checked every tick a cell turns blocked).  if the problem is
         close ahead we replan immediately, farther away we allow a few ticks' delay to save compute.
       - blocked cells turned free (a wall was removed, 09 / 11): a shorter path may now exist.
       - periodically, and whenever we drift far off the path.
     a new path replaces a still-valid one only if it is clearly shorter (hysteresis), so we don't flip between two
     near-equal routes.  we never stop to replan: planning happens inside the same tick.
     if no path to the goal exists (11), we drive to the reachable cell closest to the goal, park there watching the
     blocking wall, and keep replanning until an opening appears.

  determinism: nothing here is random and all iteration orders are fixed, so runs are deterministic.

  what is missing / known limits:
    - A* replans from scratch (no D* Lite).  fine at this grid size (~10k cells), but it is the main cost per replan.
    - the smoother is greedy (not globally optimal any-angle like Theta*), so some paths are a little longer than needed.
    - corner speeds come from a geometric model tuned against these scenarios, not a full trajectory optimisation.
    - exploration is purely optimistic: a dead end deeper than the scan range (06) is only discovered by driving in.
    - the soft wall cost trades a little path length for safety margin; its weights were tuned on these scenarios.
"""

import heapq, math

UNKNOWN, FREE, BLOCKED = 0, 1, 2
CH_BLOCKED = ord("#")   # the map is also kept as scan characters: "?" unknown, "." free, "#" blocked

# clearance rings: class k means the nearest blocked cell (or field edge) is RINGS[k] <= d < RINGS[k+1] from the cell centre.
# class 0 is hard inflation (impassable).  class NR means "at least RINGS[NR] away" (open space).
RINGS = (0.0, 0.33, 0.36, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80)
NR = len(RINGS) - 1
PENALTY = (0.0, 2.0, 1.2, 0.6, 0.35, 0.15, 0.06, 0.02, 0.0)   # extra A* cost per class (class 0 is never entered)
UNKNOWN_PENALTY = 0.0   # unknown space is planned exactly like free space (see top comment, point 4)

NEED_SMOOTH = 0.33   # clearance a shortcut segment must keep (m)
NEED_VALID = 0.31    # clearance below which an existing path counts as broken (collision is < 0.30)
HALF_DIAG = 0.0708   # half a cell diagonal: max distance from a cell centre to any point in the cell

ALPHA = 0.08         # pose filter gain toward the measurement
A_BRAKE = 2.5        # deceleration assumed by the speed profile (below a_max = 4 to leave room for steering)
A_TURN = 3.0         # lateral acceleration on arcs.  combined with A_BRAKE: sqrt(3^2 + 2.5^2) = 3.9 < a_max = 4
A_ACCEL = 2.5        # forward acceleration assumed by the speed profile
TRACK_ALLOW = 0.01   # tracking error we reserve at each corner (m)
KP = 6.0             # path-tracking feedback gain (1/s): 1 cm of error -> 6 cm/s of correction

REPLAN_EVERY = 50    # ticks between routine replans
WATCH_EVERY = 25     # ticks between replans while parked with no path to the goal
URGENT_DIST = 1.2    # a broken path this close ahead (m) is replanned immediately
OFF_PATH = 0.25      # replan if we drift this far from the path (m)

SQRT2 = math.sqrt(2.0)
MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))

def ring_of(d:float) -> int:
  """clearance class for a distance d (m)."""
  for k in range(NR):
    if d < RINGS[k + 1]: return k
  return NR

def clamp_norm(x:float, y:float, limit:float) -> tuple[float, float]:
  """scale (x, y) down to length `limit` if longer.  same rule the sim uses for speed and acceleration."""
  n = math.hypot(x, y)
  return (x, y) if n <= limit else (x * limit / n, y * limit / n)

class Agent:
  def __init__(self, cfg:dict):
    """cfg keys: width_m, height_m, resolution, robot_radius, v_max, a_max, dt, sense_cells, goal_tol, goal (x, y)."""
    self.cfg = cfg
    self.res = cfg["resolution"]
    self.W, self.H = round(cfg["width_m"] / self.res), round(cfg["height_m"] / self.res)
    self.Wm, self.Hm = self.W * self.res, self.H * self.res
    self.v_max, self.a_max, self.dt = cfg["v_max"], cfg["a_max"], cfg["dt"]
    self.goal = tuple(cfg["goal"])
    N = self.W * self.H

    # --- map ---
    self.occ = bytearray(N)                                          # UNKNOWN / FREE / BLOCKED per cell
    self.rowc = [bytearray(b"?" * self.W) for _ in range(self.H)]   # same map as scan characters, for fast row compares

    # --- clearance rings ---
    # offsets: every (dx, dy) whose cell-centre-to-cell-rectangle distance falls in some ring below RINGS[NR].
    reach = math.ceil(RINGS[NR] / self.res) + 1
    self.offsets = []
    for dy in range(-reach, reach + 1):
      for dx in range(-reach, reach + 1):
        d = math.hypot(max(abs(dx) - 0.5, 0.0), max(abs(dy) - 0.5, 0.0)) * self.res
        if d < RINGS[NR]: self.offsets.append((dx, dy, ring_of(d)))
    self.counts = [bytearray(N) for _ in range(NR)]   # counts[k][i]: blocked cells in ring k around cell i
    # base class from the field edges alone (walls outside the field never change)
    self.base = bytearray(N)
    for cy in range(self.H):
      for cx in range(self.W):
        xc, yc = (cx + 0.5) * self.res, (cy + 0.5) * self.res
        self.base[cy * self.W + cx] = ring_of(min(xc, self.Wm - xc, yc, self.Hm - yc))
    self.cls = bytearray(self.base)   # current clearance class per cell = min(edge class, first non-empty ring)

    # --- pose filter ---
    self.est = None
    self.vel = (0.0, 0.0)

    # --- path state ---
    self.path = None     # smoothed waypoints [(x, y), ...], path[0] = our position when planned
    self.cum = []        # cumulative length at each waypoint
    self.vlim = []       # speed limit at each waypoint from the profile
    self.seg = 0         # index of the segment we are on
    self.watch = False   # True when no path to the goal exists and we are parked watching
    self.tick = 0
    self.last_plan = 0
    self.dirty = False   # cells turned blocked and the path has not been re-checked (or replaced) since
    self.removed = False # a cell turned free since the last plan

  # ------------------------------------------------------------------ main loop

  def step(self, pose:tuple[float, float], scan:tuple[int, int, list[str]]) -> tuple[float, float]:
    """
    called once per tick.

    pose: (x, y) metres from SLAM, ~2 cm gaussian noise.
    scan: (cx0, cy0, rows) -- a (2*sense_cells+1)^2 window of '#'/'.' around the robot. rows[j][i] is cell (cx0+i, cy0+j).
          everything in the window is observed, nothing outside it is.
          the window origin comes from the noisy pose, so walls can land one cell off between scans.
    returns: (vx, vy) world-frame velocity command in m/s. sim clamps speed and acceleration.
    """
    self._update_pose(pose)
    self._integrate_scan(scan)
    self._maybe_replan()
    cmd = self._follow()
    # replay the sim's own clamping so we know the velocity we will actually have next tick (used by the pose filter)
    cx, cy = clamp_norm(cmd[0], cmd[1], self.v_max)
    dvx, dvy = clamp_norm(cx - self.vel[0], cy - self.vel[1], self.a_max * self.dt)
    self.vel = (self.vel[0] + dvx, self.vel[1] + dvy)
    self.tick += 1
    return cmd

  def debug(self) -> dict:
    """
    optional, for `harness.py --viz` only.
    keys: blocked (cells), free (cells), path ([(x, y), ...]).
    """
    W = self.W
    blocked = [(i % W, i // W) for i in range(len(self.cls)) if self.cls[i] == 0]
    free = [(i % W, i // W) for i in range(len(self.occ)) if self.occ[i] == FREE]
    return {"blocked": blocked, "free": free, "path": self.path or []}

  # ------------------------------------------------------------------ pose

  def _update_pose(self, pose:tuple[float, float]):
    """complementary filter: predict with our known velocity, nudge toward the noisy measurement."""
    if self.est is None:
      self.est = (float(pose[0]), float(pose[1]))
      return
    px, py = self.est[0] + self.vel[0] * self.dt, self.est[1] + self.vel[1] * self.dt
    a = max(ALPHA, 1.0 / (self.tick + 1))   # early on, a running average of all measurements so far: converges fast
    self.est = (px + a * (pose[0] - px), py + a * (pose[1] - py))

  # ------------------------------------------------------------------ map

  def _origins(self, S:int) -> list[tuple[int, int]]:
    """candidate scan origins from the filtered pose.  two per axis only when we sit within a few mm of a cell edge."""
    e = 0.006
    xs = sorted({math.floor((self.est[0] + d) / self.res) for d in (-e, 0.0, e)})
    ys = sorted({math.floor((self.est[1] + d) / self.res) for d in (-e, 0.0, e)})
    main = (math.floor(self.est[0] / self.res) - S, math.floor(self.est[1] / self.res) - S)
    cands = [(x - S, y - S) for y in ys for x in xs]
    cands.remove(main)
    return [main] + cands   # the filtered-pose origin first, so it wins ties

  def _row_span(self, ox:int, n:int) -> tuple[int, int]:
    """columns [i0, i1) of a scan row that fall inside the field, for a scan whose origin column is ox."""
    return max(0, -ox), min(n, self.W - ox)

  def _agreement(self, rows:list[str], ox:int, oy:int) -> int:
    """how many scan rows match the map exactly if the scan origin were (ox, oy).  cheap alignment score."""
    n = len(rows)
    i0, i1 = self._row_span(ox, n)
    score = 0
    for j in range(n):
      cy = oy + j
      if 0 <= cy < self.H and i0 < i1 and self.rowc[cy][ox + i0:ox + i1] == rows[j][i0:i1].encode(): score += 1
    return score

  def _integrate_scan(self, scan:tuple[int, int, list[str]]):
    """write the scan into the map, choosing the origin as described in the top comment (point 2)."""
    rows = scan[2]
    n = len(rows)
    S = (n - 1) // 2
    cands = self._origins(S)
    ox, oy = cands[0]
    if len(cands) > 1:
      best = self._agreement(rows, ox, oy)
      for cx, cy in cands[1:]:
        s = self._agreement(rows, cx, cy)
        if s > best: best, ox, oy = s, cx, cy
    i0, i1 = self._row_span(ox, n)
    if i0 >= i1: return
    W = self.W
    for j in range(n):
      cy = oy + j
      if not 0 <= cy < self.H: continue
      seg = rows[j][i0:i1].encode()
      mrow = self.rowc[cy]
      x0 = ox + i0
      if mrow[x0:x0 + len(seg)] == seg: continue   # nothing changed on this row (the common case)
      for k, ch in enumerate(seg):
        if mrow[x0 + k] == ch: continue
        i = cy * W + x0 + k
        old = self.occ[i]
        mrow[x0 + k] = ch
        if ch == CH_BLOCKED:
          self.occ[i] = BLOCKED
          self._stamp(x0 + k, cy, +1)
          self.dirty = True
        else:
          self.occ[i] = FREE
          if old == BLOCKED:
            self._stamp(x0 + k, cy, -1)
            self.removed = True

  def _stamp(self, bx:int, by:int, delta:int):
    """add (+1) or remove (-1) blocked cell (bx, by) from the clearance-ring counters of all nearby cells."""
    W, H, counts, cls = self.W, self.H, self.counts, self.cls
    for dx, dy, k in self.offsets:
      x, y = bx + dx, by + dy
      if not (0 <= x < W and 0 <= y < H): continue
      i = y * W + x
      if delta > 0:
        counts[k][i] += 1
        if k < cls[i]: cls[i] = k
      else:
        counts[k][i] -= 1
        if counts[k][i] == 0 and cls[i] == k:   # that was the closest ring: find the next non-empty one
          c = self.base[i]
          for kk in range(k + 1, min(c, NR)):
            if counts[kk][i]:
              c = kk
              break
          cls[i] = c

  # ------------------------------------------------------------------ clearance queries

  def _cell(self, x:float, y:float) -> tuple[int, int]:
    return math.floor(x / self.res), math.floor(y / self.res)

  def _clr(self, x:float, y:float) -> float:
    """
    distance (m) from point (x, y) to the nearest blocked cell or field edge, capped around 0.6 m.
    fast path: the cell's ring gives a lower bound (ring edge minus half a cell diagonal); if that is already
    comfortably large we return it.  otherwise we measure exactly against blocked cells within 0.6 m.
    """
    cx, cy = self._cell(x, y)
    if not (0 <= cx < self.W and 0 <= cy < self.H): return -1.0
    lb = RINGS[self.cls[cy * self.W + cx]] - HALF_DIAG
    if lb >= 0.45: return lb
    res = self.res
    best = min(x, self.Wm - x, y, self.Hm - y, 0.6)
    n = 6
    xlo, xhi = max(cx - n, 0), min(cx + n + 1, self.W)
    for yy in range(max(cy - n, 0), min(cy + n + 1, self.H)):
      row = self.rowc[yy]
      if CH_BLOCKED not in row[xlo:xhi]: continue   # skip rows with no walls quickly
      ddy = max(yy * res - y, 0.0, y - (yy + 1) * res)
      if ddy >= best: continue
      for xx in range(xlo, xhi):
        if row[xx] == CH_BLOCKED:
          d = math.hypot(max(xx * res - x, 0.0, x - (xx + 1) * res), ddy)
          if d < best: best = d
    return best

  def _los(self, p:tuple[float, float], q:tuple[float, float], need:float) -> bool:
    """
    True if every point on segment p->q keeps at least `need` clearance.
    clearance changes at most 1 m per metre travelled, so after a point with clearance c we can safely skip ahead
    (c - need) without missing anything.  this makes the check exact but cheap in open space.
    """
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    t = 0.0
    while True:
      f = t / L if L > 1e-9 else 0.0
      c = self._clr(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
      if c < need: return False
      if t >= L: return True
      t = min(L, t + max(c - need, 0.005))

  # ------------------------------------------------------------------ planning

  def _snap(self, i:int) -> int:
    """nearest passable cell to cell i (BFS), used when the start or goal sits inside hard inflation."""
    if self.cls[i] != 0: return i
    W, H = self.W, self.H
    seen = {i}
    q = [i]
    for cur in q:   # q grows while we iterate: plain BFS
      x, y = cur % W, cur // W
      for dx, dy, _ in MOVES[:4]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
          j = ny * W + nx
          if j in seen: continue
          if self.cls[j] != 0: return j
          seen.add(j)
          q.append(j)
    return i

  def _astar(self, s:int, g:int) -> tuple[list[int], bool]:
    """
    8-connected A* from cell s to cell g over passable cells (class != 0), with soft clearance costs.
    returns (cells, reached).  if g is unreachable, returns the path to the reachable cell closest to g instead.
    """
    W, H, cls = self.W, self.H, self.cls
    gx, gy = g % W, g // W
    def h(i:int) -> float:
      dx, dy = abs(i % W - gx), abs(i // W - gy)
      return dx + dy + (SQRT2 - 2.0) * min(dx, dy)   # octile distance: admissible, penalties only add cost
    N = W * H
    dist = [math.inf] * N
    parent = [-1] * N
    closed = bytearray(N)
    dist[s] = 0.0
    heap = [(h(s), s)]
    best, best_h = s, h(s)
    while heap:
      _, i = heapq.heappop(heap)
      if closed[i]: continue
      closed[i] = 1
      hi = h(i)
      if hi < best_h: best, best_h = i, hi
      if i == g: break
      x, y = i % W, i // W
      di = dist[i]
      for dx, dy, step in MOVES:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H): continue
        j = ny * W + nx
        cj = cls[j]
        if cj == 0 or closed[j]: continue
        # corner rule: a diagonal needs both orthogonal neighbours passable, so we never squeeze between
        # two blocked cells that only touch at a corner
        if dx and dy and (cls[i + dx] == 0 or cls[i + dy * W] == 0): continue
        nd = di + step * (1.0 + PENALTY[cj] + (UNKNOWN_PENALTY if self.occ[j] == UNKNOWN else 0.0))
        if nd < dist[j]:
          dist[j] = nd
          parent[j] = i
          heapq.heappush(heap, (nd + h(j), j))
    reached = bool(closed[g])
    end = g if reached else best
    cells = [end]
    while cells[-1] != s: cells.append(parent[cells[-1]])
    cells.reverse()
    return cells, reached

  def _plan(self) -> tuple[list[tuple[float, float]], bool]:
    """plan from the current estimate to the goal (or the closest reachable point).  returns (smoothed path, reached)."""
    W, res = self.W, self.res
    sx, sy = self._cell(*self.est)
    sx, sy = min(max(sx, 0), W - 1), min(max(sy, 0), self.H - 1)
    s0 = sy * W + sx
    s = self._snap(s0)
    gx, gy = self._cell(*self.goal)
    g = self._snap(gy * W + gx)
    cells, reached = self._astar(s, g)
    pts = [self.est]
    if s != s0: pts.append(((s % W + 0.5) * res, (s // W + 0.5) * res))   # step out of the inflation first
    pts += [((c % W + 0.5) * res, (c // W + 0.5) * res) for c in cells[1:]]
    return self._smooth(pts), reached

  def _smooth(self, pts:list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    string pulling: from each anchor, extend to the farthest waypoint still in line of sight with NEED_SMOOTH
    clearance.  segments leaving our own position use a slightly lower requirement when we start close to a wall
    (04), since moving away from the wall is always safe.  an adjacent raw step is always accepted.
    """
    if len(pts) <= 2: return pts
    out = [pts[0]]
    need0 = min(NEED_SMOOTH, max(NEED_VALID, self._clr(*pts[0]) - 0.005))
    i = 0
    while i < len(pts) - 1:
      need = need0 if i == 0 else NEED_SMOOTH
      j = i + 1
      while j + 1 < len(pts) and self._los(pts[i], pts[j + 1], need): j += 1
      out.append(pts[j])
      i = j
    return out

  def _set_path(self, path:list[tuple[float, float]], stop_at_end:bool):
    """
    turn the smoothed waypoints into a drivable curve plus a speed profile (top comment, point 6).

    each corner is replaced by a circular arc tangent to both legs.  a robot at speed v with lateral acceleration
    A_TURN can follow a radius v^2 / A_TURN, so the arc radius sets the corner speed.  the radius is the largest that
      - fits on the legs (each interior leg is shared by two corners, so each corner may use half of it),
      - cuts inside the corner by no more than the room the corner has: R * (1/cos(theta/2) - 1) <= room,
      - and is verified: every arc sample must keep NEED_VALID + TRACK_ALLOW clearance, else the radius is halved.
    the curve is then sampled every ~5 cm into self.path, with a speed cap per sample from its curvature.
    """
    n = len(path)
    legs = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    dense, rad = [path[0]], [math.inf]   # sampled curve and the curvature radius at each sample
    cur = path[0]   # where the straight part of the current leg starts (end of the previous arc)
    for i in range(1, n - 1):
      (px, py), (vx, vy), (nx, ny) = path[i - 1], path[i], path[i + 1]
      la, lb = legs[i - 1], legs[i]
      if la < 1e-9 or lb < 1e-9: continue
      ua, ub = ((vx - px) / la, (vy - py) / la), ((nx - vx) / lb, (ny - vy) / lb)
      cross = ua[0] * ub[1] - ua[1] * ub[0]
      theta = math.acos(max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1])))   # turning angle
      if theta < 1e-3: continue
      half = theta / 2.0
      arc = []
      R = 0.0
      if theta < 3.0:   # a near U-turn gets no arc: the profile slows to almost zero at the sharp corner instead
        tan_h, k = math.tan(half), 1.0 / math.cos(half) - 1.0
        room = max(self._clr(vx, vy) - NEED_VALID - TRACK_ALLOW, 0.0)
        t_avail = min(la if i == 1 else la / 2.0, lb if i == n - 2 else lb / 2.0)
        t_avail = min(t_avail, math.hypot(vx - cur[0], vy - cur[1]))   # never start before the previous arc ended
        R = min(room / k, t_avail / tan_h, self.v_max ** 2 / A_TURN)
      while R > 0.005:
        arc = self._arc((vx, vy), ua, ub, cross, half, R)
        if all(self._clr(x, y) >= NEED_VALID + TRACK_ALLOW for x, y in arc): break
        R *= 0.5
        arc = []
      if not arc:   # no arc fits: keep the sharp corner (the speed profile then nearly stops there)
        self._line(dense, rad, cur, (vx, vy))
        rad[-1] = 0.0
        cur = (vx, vy)
        continue
      self._line(dense, rad, cur, arc[0])
      for p in arc[1:]:
        dense.append(p)
        rad.append(R)
      cur = arc[-1]
    self._line(dense, rad, cur, path[-1])

    m = len(dense)
    cum = [0.0]
    for a, b in zip(dense, dense[1:]): cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    v = [min(self.v_max, math.sqrt(A_TURN * r)) for r in rad]
    v[-1] = 0.0 if stop_at_end else self.v_max   # the goal needs no stop: the run ends inside the tolerance
    for i in range(m - 2, -1, -1):   # backward pass: always able to brake to what comes next
      v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * A_BRAKE * (cum[i + 1] - cum[i])))
    v[0] = min(v[0], max(math.hypot(*self.vel), 0.1) + 0.5)
    for i in range(1, m):            # forward pass: don't plan to speed up faster than we can while turning
      v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * A_ACCEL * (cum[i] - cum[i - 1])))
    self.path, self.cum, self.vlim, self.seg = dense, cum, v, 0
    self.last_plan = self.tick
    self.removed = self.dirty = False

  def _arc(self, V:tuple[float, float], ua:tuple[float, float], ub:tuple[float, float], cross:float, half:float,
           R:float) -> list[tuple[float, float]]:
    """points (~5 cm apart) of the arc of radius R tangent to the legs meeting at vertex V (incoming ua, outgoing ub)."""
    T = R * math.tan(half)
    ax, ay = V[0] - ua[0] * T, V[1] - ua[1] * T               # tangent point on the incoming leg
    side = 1.0 if cross > 0 else -1.0                          # +1: left turn, centre on the left
    cx, cy = ax - ua[1] * R * side, ay + ua[0] * R * side      # arc centre
    a0 = math.atan2(ay - cy, ax - cx)
    sweep = 2.0 * half * side
    steps = max(2, math.ceil(abs(sweep) * R / 0.05))
    return [(cx + R * math.cos(a0 + sweep * s / steps), cy + R * math.sin(a0 + sweep * s / steps)) for s in range(steps + 1)]

  def _line(self, dense:list, rad:list, a:tuple[float, float], b:tuple[float, float]):
    """append the straight run a->b to the sampled curve (a is already in it), ~5 cm per sample."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(1, math.ceil(L / 0.05))
    for s in range(1, steps + 1):
      dense.append((a[0] + (b[0] - a[0]) * s / steps, a[1] + (b[1] - a[1]) * s / steps))
      rad.append(math.inf)

  def _replan(self, force:bool):
    """plan a new path; adopt it if forced (current one broken) or clearly shorter than what is left of the current one."""
    path, reached = self._plan()
    new_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    if not force and self.path is not None and reached and not self.watch:
      left = self.cum[-1] - self._progress()[0]
      if new_len > left - max(0.05, 0.03 * left):
        self.last_plan = self.tick
        self.removed = False
        return
    self.watch = not reached
    self._set_path(path, stop_at_end=not reached)

  def _maybe_replan(self):
    """replan policy (top comment, point 8)."""
    if self.path is None:
      self._replan(True)
      return
    since = self.tick - self.last_plan
    if self.dirty:   # the map gained walls: is our path still safe?
      bad = self._broken_at()
      if bad == math.inf: self.dirty = False
      elif bad < URGENT_DIST or since >= 8:
        self._replan(True)
        return
    s, off = self._progress()
    if off > OFF_PATH: self._replan(True)
    elif self.removed and since >= 5: self._replan(False)
    elif since >= (WATCH_EVERY if self.watch else REPLAN_EVERY): self._replan(self.watch)

  def _broken_at(self) -> float:
    """distance ahead (m) to the first point where the path loses NEED_VALID clearance, or inf if it is fine."""
    s, _ = self._progress()
    start = s + 0.1   # skip the bit we are standing on (we may start close to a wall, 04)
    path, cum = self.path, self.cum
    for k in range(self.seg, len(path) - 1):
      if cum[k + 1] <= start: continue
      a = self._point_at(max(start, cum[k]))
      if not self._los(a, path[k + 1], NEED_VALID): return max(cum[k], start) - s
    return math.inf

  # ------------------------------------------------------------------ following

  def _point_at(self, s:float) -> tuple[float, float]:
    """point at arc length s along the path (clamped to the ends)."""
    path, cum = self.path, self.cum
    if s <= 0.0: return path[0]
    if s >= cum[-1]: return path[-1]
    k = self.seg
    while k > 0 and cum[k] > s: k -= 1
    while cum[k + 1] < s: k += 1
    f = (s - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
    return (path[k][0] + (path[k + 1][0] - path[k][0]) * f, path[k][1] + (path[k + 1][1] - path[k][1]) * f)

  def _progress(self) -> tuple[float, float]:
    """project our position onto the path near the current segment.  returns (arc length, distance off the path)."""
    path, cum = self.path, self.cum
    if len(path) < 2: return 0.0, math.hypot(self.est[0] - path[0][0], self.est[1] - path[0][1])
    ex, ey = self.est
    best = (math.inf, self.seg, 0.0)
    for k in range(self.seg, min(self.seg + 25, len(path) - 1)):   # ~1.2 m of curve ahead
      (ax, ay), (bx, by) = path[k], path[k + 1]
      dx, dy = bx - ax, by - ay
      L2 = dx * dx + dy * dy
      t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((ex - ax) * dx + (ey - ay) * dy) / L2))
      d = math.hypot(ax + dx * t - ex, ay + dy * t - ey)
      if d < best[0] - 1e-9: best = (d, k, t)
    d, k, t = best
    self.seg = k
    return cum[k] + t * (cum[k + 1] - cum[k]), d

  def _follow(self) -> tuple[float, float]:
    """
    tracking controller: velocity = speed * path tangent (feedforward) + KP * offset back onto the path (feedback).
    unlike chasing a lookahead point (pure pursuit), this does not cut inside arcs, so the robot drives the curve that
    was checked for clearance.  the accel-limited sim can execute it because the profile keeps turning below A_TURN.
    """
    path, cum, vlim = self.path, self.cum, self.vlim
    ex, ey = self.est
    if len(path) < 2:
      ex_to = (path[0][0] - ex, path[0][1] - ey)
      return (0.0, 0.0) if math.hypot(*ex_to) < 0.02 else clamp_norm(ex_to[0] * 3.0, ex_to[1] * 3.0, 0.5)
    s, _ = self._progress()
    k = self.seg
    # speed at our spot on the profile: able to brake to the next sample, and not faster than the accel ramp allows
    v = min(self.v_max, math.sqrt(vlim[k + 1] ** 2 + 2.0 * A_BRAKE * max(cum[k + 1] - s, 0.0)),
            math.sqrt(vlim[k] ** 2 + 2.0 * A_ACCEL * max(s - cum[k], 0.0)))
    if self.watch:   # parking at the watch point: settle gently instead of chasing it
      d_end = math.hypot(path[-1][0] - ex, path[-1][1] - ey)
      if d_end < 0.02: return (0.0, 0.0)
      v = min(v, 3.0 * d_end)
    # feedforward: path direction half a tick ahead (the command moves us during the coming tick)
    sf = min(s + 0.5 * v * self.dt, cum[-1] - 1e-6)
    j = k
    while j < len(path) - 2 and cum[j + 1] <= sf: j += 1
    tx, ty = path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1]
    n = math.hypot(tx, ty)
    if n < 1e-9: return (0.0, 0.0)
    # feedback: pull toward our projection on the path (this error is sideways by construction)
    qx, qy = self._point_at(s)
    fx, fy = clamp_norm(KP * (qx - ex), KP * (qy - ey), 0.5)
    return (v * tx / n + fx, v * ty / n + fy)
