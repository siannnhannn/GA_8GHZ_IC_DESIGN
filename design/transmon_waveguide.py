import math
import numpy as np
from scipy.spatial import cKDTree
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

ARC_POINTS = 33   # points per rounded corner


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _fillet_path(points, r, n_arc=ARC_POINTS, tol=1e-9):
    """Round every corner of a polyline with a circular arc of radius r (any angle).

    Raises if an arc does not fit between its neighbours (nothing is silently left
    sharp or allowed to overlap)."""
    # points are (x, y) or (x, y, radius); a missing radius means `r`
    arr = np.array([[q[0], q[1], q[2] if len(q) > 2 else r] for q in points], dtype=float)
    keep = [0] + [i for i in range(1, len(arr)) if np.linalg.norm(arr[i, :2] - arr[i - 1, :2]) > tol]
    arr = arr[keep]
    pts, rad = arr[:, :2], arr[:, 2]
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    u = d / seg[:, None]
    n = len(pts)
    phi = np.zeros(n)          # turning angle at each vertex
    t = np.zeros(n)            # tangent length of the arc at each vertex
    for i in range(1, n - 1):
        phi[i] = np.arccos(np.clip(u[i - 1] @ u[i], -1.0, 1.0))
        if phi[i] > np.pi - 1e-6:
            raise ValueError(f"Path reverses on itself at {tuple(pts[i])}.")
        if phi[i] > 1e-9:
            t[i] = rad[i] * np.tan(phi[i] / 2)
    short = np.where(seg < t[:-1] + t[1:] - 1e-9)[0]
    if len(short):
        i = int(short[0])
        raise ValueError(f"Segment {tuple(pts[i])} -> {tuple(pts[i + 1])} is shorter than the "
                         "fillets at its ends need. Reduce `fillet` or lengthen the "
                         "routing distances (pitch, leg_length, bump_height, ...).")
    out = [pts[0]]
    for i in range(1, n - 1):
        if phi[i] <= 1e-9:
            out.append(pts[i])
            continue
        p1 = pts[i] - u[i - 1] * t[i]
        s = np.sign(u[i - 1][0] * u[i][1] - u[i - 1][1] * u[i][0])
        c = p1 + rad[i] * s * np.array([-u[i - 1][1], u[i - 1][0]])
        a0 = np.arctan2(p1[1] - c[1], p1[0] - c[0])
        a = a0 + s * np.linspace(0.0, phi[i], n_arc)
        out.extend(c + rad[i] * np.column_stack([np.cos(a), np.sin(a)]))
    out.append(pts[-1])
    return np.array(out)


def _arclength(coords):
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _s_at_point(coords, pt):
    """Arc-length position of the point on the polyline closest to `pt`."""
    pt = np.asarray(pt, dtype=float)
    cum = _arclength(coords)
    best_d, best_s = np.inf, 0.0
    for k in range(len(coords) - 1):
        a, ab = coords[k], coords[k + 1] - coords[k]
        L2 = ab @ ab
        t = 0.0 if L2 == 0 else float(np.clip((pt - a) @ ab / L2, 0.0, 1.0))
        d = np.linalg.norm(a + t * ab - pt)
        if d < best_d:
            best_d, best_s = d, cum[k] + t * np.sqrt(L2)
    return best_s


def _turn_t(r, d_in, d_out):
    """Tangent length of a radius-r fillet between two unit directions."""
    phi = np.arccos(np.clip(np.dot(d_in, d_out), -1.0, 1.0))
    return r * np.tan(phi / 2)


def _self_clearance(coords, min_dist, step=0.0025):
    """Smallest distance between parts of the path that are far apart along the path."""
    cum = _arclength(coords)
    s = np.arange(0.0, cum[-1], step)
    xy = np.column_stack([np.interp(s, cum, coords[:, 0]), np.interp(s, cum, coords[:, 1])])
    tree = cKDTree(xy)
    worst = np.inf
    skip = 3 * min_dist                                  # neighbours along the path don't count
    for i, j in tree.query_pairs(min_dist, output_type="ndarray"):
        if abs(s[j] - s[i]) > skip:
            worst = min(worst, float(np.linalg.norm(xy[i] - xy[j])))
    return worst


def _rows(y_first, n, pitch, x_left, x_right, x_final):
    """Vertices for n horizontal rows stacked downward from y_first.
    Row 0 heads right from the vertex the caller already added; n must be even so the
    last row heads left, ending at x_final."""
    v = []
    for k in range(n):
        y = y_first - k * pitch
        x = x_right if k % 2 == 0 else (x_left if k < n - 1 else x_final)
        v.append((x, y))
        if k < n - 1:
            v.append((x, y - pitch))
    return v


def _layout(g, n, WA, WB):
    """Waypoints of the whole waveguide (corner vertices, before filleting). Right half:

      chip edge (top) -> along the top -> down a lane parallel to lane x_A -> in along a
      row parallel to the row leaving run 1 -> leg perpendicular to the upper-right side ->
      90 deg corner onto that side (run 1) -> 90 deg corner off it -> perpendicular leg ->
      row out to lane x_A -> up to block A (upper right) -> rows, the last one runs back in
      to a leg perpendicular to the lower-right side -> 90 deg corner onto it (run 2) ->
      90 deg corner off it -> leg -> row 0 of block B (lower right) -> rows -> lane x_B ->
      feeder under the qubit -> bump up to the bottom run (run 3).

    The left half is the mirror image, traversed in reverse; it stops (open) at the end
    of the left slant run."""
    pitch, qx, r, rc = g["pitch"], g["qx"], g["r"], g["rc"]
    x_Al = x_Bl = g["x_blk"]
    x_A, x_B = x_Al - pitch, x_Bl - pitch
    x_f = x_A - g["feed_spacing"]                          # feed lane, parallel to lane x_A
    y_top = g["Qin"][1] + (n - 1) * pitch                  # first row of block A
    y_feed = g["start_y"] if g["start_y"] is not None else y_top + g["feed_clear"]

    R = [(g["start_x"], y_feed), (x_f, y_feed, r), (x_f, g["yf2"], r), (*g["Q"], r),
         (*g["W1"], rc), (*g["W2"], rc), (*g["Q2"], r), (x_A, g["Q2"][1], r), (x_A, y_top, r)]
    R += _rows(y_top, n, pitch, x_Al, x_Al + WA, g["Qin"][0])   # last vertex is Qin
    R += [(*g["Wa"], rc), (*g["Wb"], rc), (*g["Qout"], r)]
    R += _rows(g["Qout"][1], n, pitch, x_Bl, x_Bl + WB, x_B)
    R += [(x_B, g["y_h"], r), (g["x_p3"], g["y_h"], rc), (g["x_p3"], g["y_b"], rc), (qx, g["y_b"])]
    M = [(2 * qx - q[0], q[1], *q[2:]) for q in reversed(R)][1:]
    # left end: stop at the end of the left slant run (drop the mirrored feed vertices)
    M = M[:-5] + [(2 * qx - g["P1"][0], g["P1"][1])]
    return R + M


class GiantAtomWaveguide(QComponent):
    """Giant-atom waveguide capacitively coupled to all five sides of the pentagon transmon
    (`myTransmon`).

    Qubit orientation 90 (lead up). Coupled sides, in the order the waveguide meets them:

        upper-right side -> lower-right side -> base -> lower-left side -> upper-left side

    On each side the waveguide runs parallel to the metal for `coupling_length` at a fixed
    edge-to-edge distance `coupling_gap`; every run is centred on its side. Between
    consecutive runs the waveguide has `target_length` (run centre to run centre), taken up
    by a serpentine of horizontal rows.

    Layout: the four serpentine blocks sit beside the qubit, one above and one below the
    qubit's level on each side. The right end of the waveguide comes in along the top of
    the chip to `start_x`; the left end is open and stops right after the left slant run.
    The layout is mirror-symmetric about x = qubit_x. Bends have radius `fillet`, except the 90 deg corners right next to the coupled runs
    (radius `coupling_fillet`). Every run is entered and left at 90 deg through a leg
    perpendicular to its side.
    """

    default_options = Dict(
        # --- qubit geometry (match the myTransmon options)
        qubit_x="0mm",
        qubit_y="-0.6mm",             # y of the neck/pad junction (myTransmon pos_y)
        qubit_side_length="100um",   # myTransmon pad_side_length
        qubit_trace_width="25um",    # myTransmon trace_width (neck width)
        qubit_lead_length="350um",   # myTransmon lead_length (only used for the clearance check)
        qubit_pad_gap="58um",        # myTransmon pad_gap (pocket around the pad; airbridges keep out of it)
        # --- coupling
        coupling_gap="10um",         # edge-to-edge distance, waveguide to qubit metal
        coupling_length="40um",      # length of every coupled run (centred on its side)
        leg_length="100um",          # legs perpendicular to the sides, leading into / out of the runs
        coupling_fillet="30um",      # radius of the 90 deg corners right next to the coupled runs
        bump_height="100um",         # height of the bump that carries the bottom run (>= 2*coupling_fillet)
        feed_spacing="100um",        # feed line runs this far from the (parallel) line under it
        # --- serpentine blocks (x measured from qubit_x)
        block_x_min="0.75mm",        # left turns of all four blocks (closest to the qubit)
        serp_x_max="2mm",          # right-most x of the blocks = start of the launchpad
        feed_clearance="150um",      # feed line runs this far above the top row of block A (only used if start_y is None)
        y_max="2.35mm",              # highest the feed line may reach
        y_min="-2.35mm",             # lowest the lower blocks may reach
        target_length="22mm",        # run centre -> next run centre
        rows=0,                      # 0 = auto, else force an even row count
        pitch="90um",               # row spacing, must be >= 2*fillet
        fillet="45um",               # radius of all waveguide bends except those at the coupled runs
        # --- chip / layout
        start_x="2.0979mm",             # chip edge where the right end terminates
        start_y="1.2mm",                # y of the feed line / start of the waveguide (None = feed_clearance above block A)
        trace_width="10um",
        trace_gap="6um",
        layer="5",
        chip="main",
    )
    component_metadata = Dict(short_name="waveguide", _qgeometry_table_path="True")

    def get_points(self):
        return self._waveguide_points

    def make(self):
        p = self.p
        r, w, pitch = p.fillet, p.trace_width, p.pitch
        qx, qy, L, tw = p.qubit_x, p.qubit_y, p.qubit_side_length, p.qubit_trace_width
        Lc, rc = p.coupling_length, p.coupling_fillet
        if pitch < 2 * r - 1e-12:
            raise ValueError("pitch must be >= 2*fillet")
        if p.bump_height < 2 * rc - 1e-12:
            raise ValueError("bump_height must be at least 2*coupling_fillet")
        if p.rows and (int(p.rows) != p.rows or int(p.rows) % 2):
            raise ValueError("rows must be 0 (auto) or an even integer")
        if not 0 < Lc < L:
            raise ValueError("coupling_length must be positive and shorter than the side length")

        # ---- pentagon (right half), same construction as myTransmon (orientation 90)
        a = tw / 2.0
        top_angle = np.deg2rad(36.0)
        h1 = L * np.sin(top_angle)                       # depth of the shoulders below the neck
        ys = a + L * np.cos(top_angle)                   # half-width at the shoulders
        dy = ys - L / 2.0
        h2 = math.sqrt(L ** 2 - dy ** 2)                 # depth of the lower sides
        H = h1 + h2                                      # total depth of the pentagon

        D = p.coupling_gap + w / 2.0                     # centreline-to-metal distance
        neck = np.array([qx + a, qy])
        shoulder = np.array([qx + ys, qy - h1])
        base = np.array([qx + L / 2.0, qy - H])

        x_hat = np.array([1.0, 0.0])
        u1 = (shoulder - neck) / L                       # upper side, neck -> shoulder
        n1 = np.array([-u1[1], u1[0]])                   # outward normal (up / right)
        u2 = (base - shoulder) / L                       # lower side, shoulder -> base
        n2 = np.array([-u2[1], u2[0]])                   # outward normal (right / down)

        s0, s1 = (L - Lc) / 2.0, (L + Lc) / 2.0          # run start / end along each side
        O1, O2 = neck + D * n1, shoulder + D * n2

        # every run is entered and left through a 90 deg corner (radius rc) from a leg
        # perpendicular to its side
        leg = p.leg_length
        # run 1 (upper side)
        P1, P2 = O1 + s0 * u1, O1 + s1 * u1
        W1, W2 = P1 - rc * u1, P2 + rc * u1
        Q2 = W2 + leg * n1                               # end of the exit leg -> row heading right
        # the feed comes in along a row `feed_spacing` above the row leaving run 1 and
        # meets the entry leg (the line through W1 along n1) at Q
        yf2 = Q2[1] + p.feed_spacing
        Q = W1 + (yf2 - W1[1]) / n1[1] * n1
        # run 2 (lower side)
        Pa, Pb = O2 + s0 * u2, O2 + s1 * u2
        Wa, Wb = Pa - rc * u2, Pb + rc * u2
        Qin = Wa + leg * n2                              # row heading left -> entry leg
        Qout = Wb + leg * n2                             # exit leg -> row heading right
        # run 3 (base) with the bump
        y_b = qy - H - D
        y_h = y_b - p.bump_height
        x_p3 = qx + Lc / 2.0 + rc
        x_blk = qx + p.block_x_min

        sp = w + p.trace_gap
        if p.feed_spacing < sp:
            raise ValueError("feed_spacing must be at least trace_width + trace_gap")
        if Q2[1] - Qin[1] < sp:
            raise ValueError("The row leaving the upper-right run and the row entering the "
                             "lower-right run are too close; increase leg_length.")
        if Qout[1] - y_h < sp + rc:
            raise ValueError("Feeder under the qubit is too close to the first row of the "
                             "lower blocks; increase bump_height.")

        g = dict(qx=qx, start_x=p.start_x, start_y=p.start_y, pitch=pitch, r=r, rc=rc, P1=tuple(P1),
                 Q=tuple(Q), yf2=yf2, feed_spacing=p.feed_spacing, W1=tuple(W1), W2=tuple(W2), Q2=tuple(Q2), Wa=tuple(Wa),
                 Wb=tuple(Wb), Qin=tuple(Qin), Qout=tuple(Qout), y_b=y_b, y_h=y_h,
                 x_p3=x_p3, x_blk=x_blk, feed_clear=p.feed_clearance)

        # coupled-run centres: upper R, lower R, base, lower L, upper L
        c_1 = tuple(O1 + (L / 2.0) * u1)
        c_2 = tuple(O2 + (L / 2.0) * u2)
        c_3 = (qx, y_b)
        centres = [c_1, c_2, c_3, (2 * qx - c_2[0], c_2[1]), (2 * qx - c_1[0], c_1[1])]

        # ---- size the serpentine blocks. All blocks get the same row count n; length is
        # linear in the row width (slope n), so probe at W = 1 mm and solve in closed form.
        W_cap = p.serp_x_max - p.block_x_min

        def link_length(n, WA, WB, i):
            coords = _fillet_path(_layout(g, n, WA, WB), r)
            s = [_s_at_point(coords, c) for c in centres[i:i + 2]]
            return s[1] - s[0]

        n_found = None
        for n in ([int(p.rows)] if p.rows else range(2, 400, 2)):
            try:
                WB = 1.0 + (p.target_length - link_length(n, 1.0, 1.0, 1)) / n
            except ValueError:
                continue                       # too few rows for the routing to fit
            if WB <= W_cap:
                n_found = n
                break
        if n_found is None:
            raise ValueError(f"The blocks do not fit: row width would exceed the {W_cap:.3f} mm "
                             "available. Increase serp_x_max or reduce block_x_min.")
        n = n_found
        WA = 1.0 + (p.target_length - link_length(n, 1.0, WB, 0)) / n
        if min(WA, WB) < 2 * r:
            raise ValueError("Serpentine rows would be shorter than 2*fillet.")
        if WA > W_cap + 1e-9:
            raise ValueError(f"Block A needs row width {WA:.3f} mm, above the {W_cap:.3f} mm "
                             "available; increase serp_x_max.")
        y_top = Qin[1] + (n - 1) * pitch
        y_feed = p.start_y if p.start_y is not None else y_top + p.feed_clearance
        if y_feed - y_top < sp + r:
            raise ValueError(f"start_y = {y_feed:.3f} mm leaves only {(y_feed - y_top) * 1e3:.0f} um "
                             f"above the top row of block A (y = {y_top:.3f} mm); at least "
                             f"{(sp + r) * 1e3:.0f} um is needed. Raise start_y, or use a smaller "
                             "pitch / wider blocks so fewer rows are needed.")
        y_last = Qout[1] - (n - 1) * pitch
        if y_feed > p.y_max:
            raise ValueError(f"The feed line would sit at y = {y_feed:.3f} mm, above y_max = "
                             f"{p.y_max:.3f} mm. Use a smaller pitch or wider blocks.")
        if y_last < p.y_min:
            raise ValueError(f"Block B would reach y = {y_last:.3f} mm, below y_min = "
                             f"{p.y_min:.3f} mm. Use a smaller pitch or wider blocks.")

        points = _layout(g, n, WA, WB)
        coords = _fillet_path(points, r)
        cum = _arclength(coords)

        # nothing may come closer to the qubit metal than the coupling distance
        lead = p.qubit_lead_length
        pad = draw.Polygon([(qx - a, qy + lead), (qx + a, qy + lead), (qx + a, qy),
                            tuple(shoulder), tuple(base), (2 * qx - base[0], base[1]),
                            (2 * qx - shoulder[0], shoulder[1]), (qx - a, qy)])
        dmin = draw.LineString(coords).distance(pad)
        if dmin < D - 1e-4:
            raise ValueError(f"The waveguide comes within {(dmin - w / 2) * 1e3:.1f} um of the "
                             f"qubit metal (coupling_gap = {p.coupling_gap * 1e3:.1f} um) away "
                             "from the coupled runs. Reduce coupling_length.")
        # and no two separate parts of the waveguide may come closer than trace_width + gap
        worst = _self_clearance(coords, sp - 1e-4)
        if worst < sp - 1e-4:
            raise ValueError(f"Two parts of the waveguide come within {worst * 1e3:.1f} um "
                             f"(centre to centre) of each other; {sp * 1e3:.1f} um is needed. "
                             "Reduce coupling_length or change feed_spacing / bump_height.")
        s_c = [_s_at_point(coords, c) for c in centres]

        # Places where add_airbridges() must not put a bridge (all in mm; each point is kept
        # clear by the same `keepout` radius):
        #  1. the qubit outline, so no foot lands in the pocket or on the pad;
        #  2. the corners drawn with coupling_fillet: route_airbridges places along ONE fillet
        #     radius, so there its path differs slightly from the rendered one;
        #  3. spots where two far-apart parts of the trace come within 45 um of each other
        #     (the legs around the coupled runs): a bridge foot would land on the other trace;
        #  4. straight pieces that are not at a multiple of 45 degrees (the legs perpendicular
        #     to the pentagon sides): bridge spans are snapped to 45 degree steps, so on them
        #     the span would be skewed against the trace and a foot could clip its gap.
        outline = [(qx - a, qy + lead), (qx + a, qy + lead), (qx + a, qy), tuple(shoulder),
                   tuple(base), (2 * qx - base[0], base[1]), (2 * qx - shoulder[0], shoulder[1]),
                   (qx - a, qy), (qx - a, qy + lead)]
        keep = []
        for u, v in zip(outline[:-1], outline[1:]):
            u, v = np.array(u), np.array(v)
            m = max(int(np.ceil(np.linalg.norm(v - u) / 0.025)), 1)
            keep.extend(tuple(u + (v - u) * k / m) for k in range(m))
        keep.extend(tuple(q[:2]) for q in points if len(q) > 2 and abs(q[2] - r) > 1e-9)
        fine = int(cum[-1] / 0.01) + 1
        sf = np.linspace(0.0, cum[-1], fine)
        xyf = np.column_stack([np.interp(sf, cum, coords[:, 0]), np.interp(sf, cum, coords[:, 1])])
        for i, j in cKDTree(xyf).query_pairs(0.045, output_type="ndarray"):
            if abs(sf[j] - sf[i]) > 0.15:
                keep.append(tuple((xyf[i] + xyf[j]) / 2.0))
        for u, v in zip(points[:-1], points[1:]):
            u, v = np.array(u[:2]), np.array(v[:2])
            ang = np.degrees(np.arctan2(v[1] - u[1], v[0] - u[0])) % 45.0
            if min(ang, 45.0 - ang) > 1.0:
                m = max(int(np.ceil(np.linalg.norm(v - u) / 0.025)), 1)
                keep.extend(tuple(u + (v - u) * k / m) for k in range(m + 1))
        self._airbridge_keepout_points = keep
        self._qubit_outline_points = outline

        # bookkeeping to inspect from the notebook
        self._waveguide_points = np.array([q[:2] for q in points], dtype=float)
        self.coupling_centres = centres
        self.coupling_spacings = list(np.diff(s_c))     # run 1->2, 2->3, 3->4, 4->5
        self.coupling_length = Lc
        self.rows_per_block = n
        self.row_width_block_a, self.row_width_block_b = WA, WB
        self.length_total = cum[-1]

        centerline = draw.LineString(coords)
        self.add_qgeometry("path", {"waveguide": centerline},
                           width=w, fillet=0, layer=p.layer, chip=p.chip)
        self.add_qgeometry("path", {"waveguide_etch": centerline},
                           width=w + 2 * p.trace_gap, fillet=0,
                           layer=p.layer, subtract=True, chip=p.chip)

        def pin(name, end, dirn):
            t = np.array([dirn[1], -dirn[0]])
            self.add_pin(name, input_as_norm=False, width=w,
                         points=np.array([end - t * w / 2, end + t * w / 2]))

        xy = self._waveguide_points
        d1 = xy[-1] - xy[-2]
        pin("waveguide_connect", xy[0], (1, 0))                    # right end, at the chip edge
        pin("waveguide_end", xy[-1], d1 / np.linalg.norm(d1))      # left end, open

    