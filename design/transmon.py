import numpy as np

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

from shapely.geometry import box
from shapely.geometry import JOIN_STYLE

class myTransmon(QComponent):
    """Transmon qubit component: pentagon island, readout lead and an optional SQUID.

    Inherits 'QComponent' class.

    The SQUID branches off the side of the protruding lead ('squid_offset' up the lead from
    the pad), crosses the pocket gap and lands on the ground plane. It is a loop with two
    Josephson-junction arms; each arm has a `squid_jj_gap` break bridged by a junction patch
    on its own layer (`squid_jj_layer`). With orientation 90 the lead points up and
    squid_side='left' puts the SQUID on the qubit's left.

    Default Options:
        * trace_width: width of the readout lead
        * lead_length: length of the readout lead
        * pad_side_length: side length of the pentagon pad
        * pad_gap: gap between pad / lead and ground (pocket)
        * squid_*: SQUID geometry (see below)
    """

    default_options = Dict(
        trace_width="25um",
        lead_length="350um",
        pad_side_length="100um",
        pad_gap="58um",
        orientation="90",
        # --- SQUID on the lead
        squid="False",                 # False: no SQUID
        squid_offset="150um",         # distance from the pad (start of the lead) to the SQUID centre
        squid_side="left",            # 'left' or 'right' of the lead (orientation 90)
        squid_loop_width="20um",      # outer width of the loop, along the lead
        squid_loop_height="20um",     # inner height of the loop, across the gap
        squid_wire_width="3um",       # stems, bars and arms
        squid_jj_gap="2um",           # break in each arm, bridged by a junction patch
        squid_ground_overlap="3um",   # how far the stem reaches into the ground plane
        squid_jj_layer="6",           # layer of the junction patches
    )

    def make(self):
        """Build the qgeometry for the component."""
        p = self.p
        s = p.pad_side_length
        w = p.trace_width
        w_half = w / 2.0
        lead_length = p.lead_length
        pad_gap = p.pad_gap

        #########################################################
        # Pentagon geometry, built with the lead along +x and the pentagon
        # body toward -x. The orientation (default 90) rotates it to point up.
        #
        # The top sides keep the regular-pentagon angle of 36 degrees from the
        # flat top. Their inner ends sit at y = +/- w/2, so the lead fits
        # between them without cutting into them.
        top_angle = np.deg2rad(36.0)
        h1 = s * np.sin(top_angle)                # depth of the shoulders below the top
        y_shoulder = w_half + s * np.cos(top_angle)  # half-width at the shoulders

        # The lower sides (length s) run from the shoulders to a base of length s
        dy_lower = y_shoulder - s / 2.0
        h2 = np.sqrt(s**2 - dy_lower**2)          # depth of the lower sides
        H = h1 + h2                               # total pentagon depth

        # Pentagon vertices (counterclockwise)
        top_pos = (0, w_half)
        shoulder_pos = (-h1, y_shoulder)
        base_pos = (-H, s / 2.0)
        base_neg = (-H, -s / 2.0)
        shoulder_neg = (-h1, -y_shoulder)
        top_neg = (0, -w_half)

        # Main pad: pentagon plus the lead
        transmon = draw.Polygon(
            [
                top_pos,
                shoulder_pos,
                base_pos,
                base_neg,
                shoulder_neg,
                top_neg,
                (lead_length, -w_half),
                (lead_length, w_half),
            ]
        )

        # Pocket: the pad grown by pad_gap (sharp corners), trimmed flush with
        # the end of the lead so the readout pin stays open for routing.
        margin = 2 * pad_gap
        pocket = transmon.buffer(pad_gap, join_style=JOIN_STYLE.mitre)
        pocket = pocket.intersection(
            box(-(H + margin), -(y_shoulder + margin), lead_length+margin, y_shoulder + margin)
        )

        #########################################################
        # SQUID branching off the side of the lead (local frame, before rotation)
        use_squid = str(p.squid).lower() in ("true", "1", "yes")
        if use_squid:
            if p.squid_side not in ("left", "right"):
                raise ValueError("squid_side must be 'left' or 'right'.")
            half_loop = p.squid_loop_width / 2.0
            if not (half_loop < p.squid_offset < lead_length - half_loop):
                raise ValueError("squid_offset puts the SQUID off the end of the lead.")
            island_pts, ground_pts, jj_pts = _squid_outlines(
                w, pad_gap, p.squid_offset, p.squid_side, p.squid_loop_width,
                p.squid_loop_height, p.squid_wire_width, p.squid_jj_gap,
                p.squid_ground_overlap)
            squid_island = draw.Polygon(island_pts)
            squid_ground = draw.Polygon(ground_pts)
            squid_jj_a, squid_jj_b = draw.Polygon(jj_pts[0]), draw.Polygon(jj_pts[1])

        #########################################################
        # Pin lines. Each side is listed clockwise around the pad, so every pin
        # normal points outward, away from the pad.
        main_pin_line = draw.LineString([(lead_length, w_half), (lead_length, -w_half)])

        # Pin names describe positions with the default orientation of 90
        pin_upper_left = draw.LineString([shoulder_pos, top_pos])
        pin_left = draw.LineString([base_pos, shoulder_pos])
        pin_bottom = draw.LineString([base_neg, base_pos])
        pin_right = draw.LineString([shoulder_neg, base_neg])
        pin_upper_right = draw.LineString([top_neg, shoulder_neg])

        polys = [
            main_pin_line, transmon, pocket,
            pin_upper_left, pin_left, pin_bottom, pin_right, pin_upper_right,
        ]
        if use_squid:
            polys += [squid_island, squid_ground, squid_jj_a, squid_jj_b]

        # Rotate and translate everything
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, xoff=p.pos_x, yoff=p.pos_y)
        [
            main_pin_line, transmon, pocket,
            pin_upper_left, pin_left, pin_bottom, pin_right, pin_upper_right,
        ] = polys[:8]
        if use_squid:
            squid_island, squid_ground, squid_jj_a, squid_jj_b = polys[8:]

        # Add to the qgeometry table
        self.add_qgeometry("poly", dict(transmon=transmon), layer=p.layer)
        self.add_qgeometry("poly", dict(pocket=pocket), layer=p.layer, subtract=True)
        if use_squid:
            self.add_qgeometry("poly", dict(squid_island=squid_island,
                                            squid_ground=squid_ground), layer=p.layer)
            self.add_qgeometry("poly", dict(squid_jj_a=squid_jj_a, squid_jj_b=squid_jj_b),
                               layer=p.squid_jj_layer)

        # Pins
        self.add_pin("readout", points=list(main_pin_line.coords),
                     width=w, input_as_norm=False)
        self.add_pin("upper_left", points=list(pin_upper_left.coords),
                     width=w, input_as_norm=False)
        self.add_pin("left", points=list(pin_left.coords),
                     width=w, input_as_norm=False)
        self.add_pin("bottom", points=list(pin_bottom.coords),
                     width=w, input_as_norm=False)
        self.add_pin("right", points=list(pin_right.coords),
                     width=w, input_as_norm=False)
        self.add_pin("upper_right", points=list(pin_upper_right.coords),
                     width=w, input_as_norm=False)