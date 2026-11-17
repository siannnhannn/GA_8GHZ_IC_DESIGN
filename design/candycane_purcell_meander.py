import numpy as np

from collections import OrderedDict
from qiskit_metal import Dict

from candycane_purcell import CandyCanePurcell
from coupling_claw import claw_geometry

# purcell filter and readout resonator for qubit
#Purcell filter and readout resonator for the qubits: CandyCanePurcell
#(Val's straight version, no meanders). Each line is one straight run with a
#single U-turn. Note the component's path names are the other way round from
#their job here:
#  "resonator" path = Purcell filter: from the T pad on the readout line,
#                     down, U-turn, back up (on the chip-centre side)
#  "purcell" path   = readout resonator: from resonator_connect (routed to the
#                     qubit's readout coupler), up, U-turn, down (chip-edge side)
#Frequencies from Almanakly et al., "Driven giant atoms" (Table S6): qubit A's
#resonator and filter at 9.8 GHz, qubit B's at 9.9 GHz. The resonators are
#lambda/4 (3.7115980428 mm); the filters are
#shorter (2.42 and 2.39 mm drawn) because of the capacitance of their pad by
#the readout line, as in the paper's device (see the tuner cell). The tuner
#sets up_from_turn (filter) and down_from_turn (resonator). fillet = 70um makes each U-turn 140um wide. No meanders, like
#the paper's (Fig. 2a): the resonator goes straight up from the qubit, turns
#under the readout line and comes straight back down (ends at y ~ 0.72mm);
#the filter goes straight down from its T pad, turns at y ~ 0.30mm, just above
#the qubit, and comes back up to y ~ 1.1mm. As in the paper, the resonator's hairpin is on the chip-edge
#side and the filter's (with its T pad) on the chip-centre side.
#(CandyCanePurcellMeander can add jogs: meander_jogs / filter_meander_jogs,
#both 0 here.)
#pos_x (the T pad) = qubit_x - 2*89.282um - 10.536um = 0.6327mm (towards the
#chip centre) puts resonator_connect right above each qubit's readout coupler;
#orientation = 1 for qubit A and -1 for qubit B puts the filter on the inside.

class CandyCanePurcellMeander(CandyCanePurcell):
    """CandyCanePurcell with meander jogs on the return leg of each path.
    "purcell" path (the readout resonator here): meander_jogs on its leg back
    down, towards -direction, starting meander_start below the U-turn.
    "resonator" path (the Purcell filter here): filter_meander_jogs on its leg
    back up, towards +direction, starting
    filter_meander_start above the U-turn. Each jog goes meander_width
    (filter: filter_meander_width) sideways, meander_depth along the leg and
    back; jogs are meander_gap apart. With fillet = 70um, widths, depth and
    gap must be >= 140um.

    Coupling claw (optional): set claw_component (and claw_pin) to the qubit
    and its readout pin, and the claw that wraps that pin is drawn as part of
    this component. The readout resonator ("purcell" path) then starts at the
    end of the claw's lead and runs straight up, so claw + resonator are one
    continuous piece of metal (no separate CouplingClaw or RouteStraight link).
    resonator_connect is moved to that claw/resonator junction. The claw lead
    has the resonator's width (meander_line_width). Place the component so
    resonator_connect's x (pos_x - orientation*(2*mini_rect_width +
    purcell_res_separation)) equals the qubit pin's x. The qubit must be
    created before this component."""
    default_options = Dict(
        meander_jogs="0",
        filter_meander_jogs="0",
        meander_width="140um",
        filter_meander_width="140um",
        meander_depth="140um",
        meander_gap="140um",
        meander_start="500um",
        filter_meander_start="600um",
        # coupling claw on the qubit's readout lead ("" = no claw)
        claw_component="",
        claw_pin="readout",
        claw_coupling_gap="4um",
        claw_overlap_depth="30um",
        claw_prong_width="16um",
        claw_bridge_width="12um",
    )

    def purcell(self, start_pin_x, start_pin_y, p, direction):
        self._claw_start = None
        if str(self.options.claw_component).strip():
            claw = self.make_claw(start_pin_x, start_pin_y, p)
        # rounded_centerline (below) adds the jogs while this path is drawn
        self._meander_for = ("purcell", p, direction)
        try:
            result = super().purcell(start_pin_x, start_pin_y, p, direction)
        finally:
            self._meander_for = None
        if self._claw_start is not None:
            self.add_qgeometry("poly", {"claw": claw}, layer=p.layer, chip=p.chip)
            # resonator_connect: the claw/resonator junction (same point order
            # as CandyCanePurcell's, so the same normal)
            x0, y0 = self._claw_start
            w = p.meander_line_width
            self.add_pin("resonator_connect",
                         points=self.rotate_points(
                             np.array([[x0 + w/2, y0], [x0 - w/2, y0]]), p),
                         width=w, input_as_norm=False)
        return result

    def make_claw(self, x0, y0, p):
        """Claw around the qubit's readout pin; sets self._claw_start, the
        end of its lead in this component's unrotated frame, where the readout
        resonator path starts."""
        claw, lead_end, normal = claw_geometry(
            self.design, str(self.options.claw_component).strip(),
            str(self.options.claw_pin).strip(),
            p.claw_coupling_gap, p.claw_overlap_depth, p.claw_prong_width,
            p.claw_bridge_width, p.meander_line_width)
        # undo this component's rotation to compare with the path's anchors
        a = -np.deg2rad(p.rotation)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        origin = np.array(self.rotation_origin(p))
        end = (lead_end - origin) @ rot.T + origin
        n = rot @ normal
        if n[1] < 1 - 1e-6:
            raise ValueError(f"{self.name}: the claw on {self.options.claw_component}."
                             f"{self.options.claw_pin} must point up, along the readout "
                             "resonator's first leg.")
        if abs(end[0] - x0) > 1e-4:  # 0.1um
            raise ValueError(f"{self.name}: resonator_connect (x = {x0:.6f}mm) is not "
                             f"in line with the claw (x = {end[0]:.6f}mm); move pos_x by "
                             f"{(end[0] - x0) * 1e3:.3f}um.")
        if end[1] >= y0:
            raise ValueError(f"{self.name}: the claw ends above resonator_connect.")
        self._claw_start = (x0, end[1])
        return claw

    def resonator(self, start_pin_x, start_pin_y, purcell_rect_x, purcell_rect_y, p, direction):
        self._meander_for = ("resonator", p, direction)
        try:
            return super().resonator(start_pin_x, start_pin_y, purcell_rect_x,
                                     purcell_rect_y, p, direction)
        finally:
            self._meander_for = None

    def rounded_centerline(self, anchors, radius, *args, **kwargs):
        if getattr(self, "_meander_for", None):
            path, p, direction = self._meander_for
            if path == "purcell":
                if getattr(self, "_claw_start", None) is not None:
                    # start the resonator at the claw: its first leg runs from
                    # the claw's lead straight up through resonator_connect
                    anchors = OrderedDict(anchors)
                    anchors[0] = np.array(self._claw_start, dtype=float)
                anchors = self.add_meander(anchors, p, int(p.meander_jogs), -direction,
                                           -1, p.meander_start, p.meander_width)
                # exact centerline length (straight runs + true arcs) of the
                # readout resonator, from the claw (if any) to its open end
                self._purcell_length = self.centerline_length(anchors, radius)
                # the airbridges follow these points
                self._purcell_points = self.rotate_points(
                    np.array(list(anchors.values()), dtype=float), p)
            else:
                anchors = self.add_meander(anchors, p, int(p.filter_meander_jogs), direction,
                                           1, p.filter_meander_start, p.filter_meander_width)
                self._resonator_points = self.rotate_points(
                    np.array(list(anchors.values()), dtype=float), p)
        return super().rounded_centerline(anchors, radius, *args, **kwargs)

    @staticmethod
    def centerline_length(anchors, radius):
        """Length of the filleted centerline through anchors: each corner of
        angle theta replaces 2*r*tan(theta/2) of straight line with an arc
        r*theta (exact, unlike the polyline drawn for the arcs)."""
        pts = [np.array(v, dtype=float) for v in anchors.values()]
        length = sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))
        for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
            u, v = b - a, c - b
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu == 0 or nv == 0:
                continue
            theta = np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
            length += radius * theta - 2 * radius * np.tan(theta / 2)
        return length

    def resonator_length(self):
        """Length of the readout resonator ("purcell" path) in mm, along its
        centerline from the end of the claw's lead (or resonator_connect, if
        there is no claw) to its open end."""
        return self._purcell_length

    def tune_resonator_length(self, target="3.712mm", tol="0.01um", max_iter=5):
        """Set down_from_turn (the resonator's open-ended leg) so that
        resonator_length() equals target, within tol. Rebuilds this component;
        rebuild anything built from it (e.g. UnionShapes) afterwards.
        Returns the final length in mm."""
        target = self.design.parse_value(target)
        tol = self.design.parse_value(tol)
        for _ in range(max_iter):
            error = target - self.resonator_length()
            if abs(error) <= tol:
                break
            # the open leg runs straight, so the length changes 1:1 with it
            self.options.down_from_turn = f"{self.p.down_from_turn + error:.6f}mm"
            self.rebuild()
        length = self.resonator_length()
        if abs(length - target) > tol:
            raise ValueError(f"{self.name}: readout resonator is {length:.6f}mm, "
                             f"could not tune it to {target:.6f}mm")
        return length

    def add_meander(self, anchors, p, jogs, side, along, start, width):
        """Insert jogs on the leg from anchor 2 to anchor 3; along = -1 if
        that leg runs down, +1 if it runs up; side = direction of the jogs."""
        pts = [np.array(v, dtype=float) for v in anchors.values()]
        if jogs <= 0:
            return anchors
        x_leg, y = pts[2]
        new = pts[:3]
        y += along * start
        for _ in range(jogs):
            y_next = y + along * p.meander_depth
            new += [np.array([x_leg, y]),
                    np.array([x_leg + side*width, y]),
                    np.array([x_leg + side*width, y_next]),
                    np.array([x_leg, y_next])]
            y = y_next + along * p.meander_gap
        new.append(pts[3])
        return OrderedDict(enumerate(new))