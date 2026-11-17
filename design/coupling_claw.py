
import numpy as np

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

from shapely.geometry import LineString, box
from shapely.ops import unary_union


def claw_geometry(design, ref_component, ref_pin, coupling_gap, overlap_depth,
                  prong_width, bridge_width, lead_width):
    """U-shaped coupling claw around the end of a transmon lead, in chip
    coordinates (all lengths in mm, already parsed).

    Returns (claw, lead_end, normal):
      claw     -- the claw metal (prongs + bridge + lead) as one polygon
      lead_end -- centre of the lead's far end (on the pocket boundary)
      normal   -- unit vector from the transmon lead out through the claw
    Used by CouplingClaw and by CandyCanePurcellMeander (claw_component).
    """
    ref = design.components[ref_component]
    pin = ref.pins[ref_pin]

    pin_pos = np.array(pin["middle"], dtype=float)
    normal = np.array(pin["normal"], dtype=float)
    normal = normal / np.linalg.norm(normal)

    pin_points = np.array(pin["points"], dtype=float)
    transmon_width = np.linalg.norm(pin_points[1] - pin_points[0])

    gap = coupling_gap
    # lead length so its end reaches the transmon's pocket boundary
    pad_gap = design.parse_value(ref.options.pad_gap)
    lead_length = pad_gap - gap - bridge_width

    if min(gap, overlap_depth, prong_width, bridge_width, lead_width, lead_length) <= 0:
        raise ValueError("All claw dimensions must be positive "
                         "(lead length = pad_gap - coupling_gap - bridge_width).")

    # equal clearance on left and right
    opening_width = transmon_width + 2 * gap

    # local frame: origin at the centre of the transmon lead tip, +y = pin normal
    left_prong = box(-opening_width / 2 - prong_width, -overlap_depth,
                     -opening_width / 2, gap)
    right_prong = box(opening_width / 2, -overlap_depth,
                      opening_width / 2 + prong_width, gap)
    bridge = box(-opening_width / 2 - prong_width, gap,
                 opening_width / 2 + prong_width, gap + bridge_width)
    lead_start = gap + bridge_width
    lead = box(-lead_width / 2, lead_start, lead_width / 2, lead_start + lead_length)

    claw = unary_union([left_prong, right_prong, bridge, lead])

    orientation = np.degrees(np.arctan2(normal[1], normal[0])) - 90
    claw = draw.rotate(claw, orientation, origin=(0, 0))
    claw = draw.translate(claw, xoff=pin_pos[0], yoff=pin_pos[1])

    lead_end = pin_pos + normal * (lead_start + lead_length)
    return claw, lead_end, normal


class CouplingClaw(QComponent):
    """
    U-shaped capacitive coupling claw.

    Automatically wraps around the end of a transmon lead
    using the referenced pin's position, width, and normal.

    coupling_gap controls the left, right, and tip spacing.
    overlap_depth controls how far the prongs extend back
    alongside the transmon lead.

    (CandyCanePurcellMeander can draw the same claw as part of the readout
    resonator: set its claw_component option.)
    """

    default_options = Dict(
        # Reference component
        ref_component="",
        ref_pin="readout",

        # Coupling geometry
        coupling_gap="6um",
        overlap_depth="30um",

        # Claw metal
        prong_width="8um",
        bridge_width="8um",

        # Connection lead
        lead_width="8um",
        lead_length="30um",   # unused: the lead always ends at the pocket boundary

        # Fabrication
        layer="5",
        chip="main",
    )

    component_metadata = Dict(
        short_name="CouplingClaw"
    )

    def make(self):
        p = self.p

        claw, lead_end, normal = claw_geometry(
            self.design, self.options.ref_component, self.options.ref_pin,
            p.coupling_gap, p.overlap_depth, p.prong_width, p.bridge_width,
            p.lead_width)

        self.add_qgeometry("poly", {"claw": claw}, layer=p.layer, chip=p.chip)

        # connection pin across the end of the lead
        # (same point order as before the refactor, so the pin normal is unchanged)
        side = np.array([normal[1], -normal[0]]) * p.lead_width / 2
        pin_line = LineString([lead_end - side, lead_end + side])
        self.add_pin("connect", list(pin_line.coords), width=p.lead_width)
