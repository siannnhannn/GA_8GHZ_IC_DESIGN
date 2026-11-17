import numpy as np

from collections import OrderedDict
from qiskit_metal import Dict

from candycane_purcell import CandyCanePurcell

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
    gap must be >= 140um."""
    default_options = Dict(
        meander_jogs="0",
        filter_meander_jogs="0",
        meander_width="140um",
        filter_meander_width="140um",
        meander_depth="140um",
        meander_gap="140um",
        meander_start="500um",
        filter_meander_start="600um",
    )

    def purcell(self, start_pin_x, start_pin_y, p, direction):
        # rounded_centerline (below) adds the jogs while this path is drawn
        self._meander_for = ("purcell", p, direction)
        try:
            return super().purcell(start_pin_x, start_pin_y, p, direction)
        finally:
            self._meander_for = None

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
                anchors = self.add_meander(anchors, p, int(p.meander_jogs), -direction,
                                           -1, p.meander_start, p.meander_width)
                # the airbridges follow these points
                self._purcell_points = self.rotate_points(
                    np.array(list(anchors.values()), dtype=float), p)
            else:
                anchors = self.add_meander(anchors, p, int(p.filter_meander_jogs), direction,
                                           1, p.filter_meander_start, p.filter_meander_width)
                self._resonator_points = self.rotate_points(
                    np.array(list(anchors.values()), dtype=float), p)
        return super().rounded_centerline(anchors, radius, *args, **kwargs)

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