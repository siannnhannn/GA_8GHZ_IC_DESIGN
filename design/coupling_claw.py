
import numpy as np

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

from shapely.geometry import LineString, box
from shapely.ops import unary_union


class CouplingClaw(QComponent):
    """
    U-shaped capacitive coupling claw.

    Automatically wraps around the end of a transmon lead
    using the referenced pin's position, width, and normal.

    coupling_gap controls the left, right, and tip spacing.
    overlap_depth controls how far the prongs extend back
    alongside the transmon lead.
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
        lead_length="30um",

        # Fabrication
        layer="5",
        chip="main",
    )

    component_metadata = Dict(
        short_name="CouplingClaw"
    )

    def make(self):

        p = self.p

        # ---------------------------------
        # Reference transmon pin
        # ---------------------------------

        ref_name = self.options.ref_component
        ref_pin_name = self.options.ref_pin

        ref_component = self.design.components[ref_name]
        ref_pin = ref_component.pins[ref_pin_name]

        pin_pos = np.array(
            ref_pin["middle"], dtype=float
        )

        normal = np.array(
            ref_pin["normal"], dtype=float
        )

        normal = normal / np.linalg.norm(normal)

        # ---------------------------------
        # Determine transmon lead width
        # ---------------------------------

        pin_points = np.array(
            ref_pin["points"], dtype=float
        )

        transmon_width = np.linalg.norm(
            pin_points[1] - pin_points[0]
        )

        # ---------------------------------
        # Parameters
        # ---------------------------------

        gap = p.coupling_gap
        overlap_depth = p.overlap_depth

        prong_width = p.prong_width
        bridge_width = p.bridge_width

        lead_width = p.lead_width

        # Get the transmon's surrounding pocket clearance
        pad_gap = self.design.parse_value(
            ref_component.options.pad_gap
        )

        # Calculate lead length so its end reaches the pocket boundary
        lead_length = pad_gap - gap - bridge_width

        if min(
            gap, overlap_depth, prong_width,
            bridge_width, lead_width, lead_length
        ) <= 0:
            raise ValueError(
                "All claw dimensions must be positive."
            )

        # Equal clearance on left and right
        opening_width = transmon_width + 2 * gap

        # ---------------------------------
        # Automatic position and orientation
        # ---------------------------------

        # Origin is the center of the transmon lead tip.
        pos_x, pos_y = pin_pos

        # Align local +y with the outward pin normal.
        orientation = (
            np.degrees(
                np.arctan2(normal[1], normal[0])
            ) - 90
        )

        # ---------------------------------
        # Left prong
        # ---------------------------------

        left_prong = box(
            -opening_width / 2 - prong_width,
            -overlap_depth,
            -opening_width / 2,
            gap,
        )

        # ---------------------------------
        # Right prong
        # ---------------------------------

        right_prong = box(
            opening_width / 2,
            -overlap_depth,
            opening_width / 2 + prong_width,
            gap,
        )

        # ---------------------------------
        # Top bridge
        # ---------------------------------

        bridge = box(
            -opening_width / 2 - prong_width,
            gap,
            opening_width / 2 + prong_width,
            gap + bridge_width,
        )

        # ---------------------------------
        # Connection lead
        # ---------------------------------

        lead_start = gap + bridge_width

        lead = box(
            -lead_width / 2,
            lead_start,
            lead_width / 2,
            lead_start + lead_length,
        )

        # ---------------------------------
        # Combine metal geometry
        # ---------------------------------

        claw = unary_union([
            left_prong,
            right_prong,
            bridge,
            lead,
        ])

        # ---------------------------------
        # Rotate and translate
        # ---------------------------------

        claw = draw.rotate(
            claw,
            orientation,
            origin=(0, 0),
        )

        claw = draw.translate(
            claw,
            xoff=pos_x,
            yoff=pos_y,
        )

        # ---------------------------------
        # Add geometry
        # ---------------------------------

        self.add_qgeometry(
            "poly",
            {"claw": claw},
            layer=p.layer,
            chip=p.chip,
        )

        # ---------------------------------
        # Connection pin
        # ---------------------------------

        pin_y = lead_start + lead_length

        pin_line = LineString([
            (-lead_width / 2, pin_y),
            (lead_width / 2, pin_y),
        ])

        pin_line = draw.rotate(
            pin_line,
            orientation,
            origin=(0, 0),
        )

        pin_line = draw.translate(
            pin_line,
            xoff=pos_x,
            yoff=pos_y,
        )

        self.add_pin(
            "connect",
            list(pin_line.coords),
            width=lead_width,
        )
