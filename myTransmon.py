import qiskit_metal as qm
from qiskit_metal import designs, draw, Dict
from qiskit_metal.qlibrary.core import QComponent
import numpy as np


class myTransmon(QComponent):
    """Transmon qubit component.

    Inherits 'QComponent' class.

    
    Default Options:
        * trace_width: 'cpw_width' -- Width of the transmission line attached to the transmon
        * trace_gap: 'cpw_gap' -- Gap of the transmission line
        * lead_length: '25um' -- Length of the transmission line attached to the transmon
        * pad_width: '80um' -- Width of the transmon pad
        * pad_height: '80um' -- Height of the transmon pad
        * pad_gap: '58um' -- Gap of the transmon pad
        * taper_height: '122um' -- Height of the taper from the transmon to the transmission line
        * edge_length: '100um' -- Length of the edges of the transmon
    """

    default_options = Dict(
        trace_width="10um",
        lead_length="25um",
        pad_width="80um",
        pad_height="80um",
        pad_gap="58um",
        taper_height="122um",
        edge_length="100um",
        orientation="90"
    )





    def make(self):
        """This is executed by the user to generate the qgeometry for the
        component."""

        p = self.p

        pad_width = p.pad_width
        pad_height = p.pad_height
        pad_gap = p.pad_gap
        edge_length = p.edge_length
        trace_width = p.trace_width
        trace_width_half = trace_width / 2.0
        pad_width_half = pad_width / 2.0
        lead_length = p.lead_length
        taper_height = p.taper_height
        pad_gap = p.pad_gap
        #########################################################

        # Geometry of main transmon structure
        # The shape is a polygon and we prepare this point as orientation is 90 degree


        R = edge_length / (2 * np.sin(np.pi / 5))

        angles = np.deg2rad([0, 72, 144, 216, 288])

        vertices = [
        (R * np.cos(a), R * np.sin(a))
        for a in angles
        ]

        transmon = draw.Polygon(vertices)


        

        # Geometry pocket (gap)
        # Same way applied for pocket

        
        pocket = draw.Polygon(
            [
                (0, trace_width_half + pad_gap),
                (-taper_height, pad_width_half + pad_gap),
                (-(pad_height + taper_height + pad_gap), pad_width_half + pad_gap),
                (-(pad_height + taper_height + pad_gap), -(pad_width_half + pad_gap)),
                (-taper_height, -(pad_width_half + pad_gap)),
                (0, -(trace_width_half + pad_gap)),
                (lead_length, -(trace_width_half + pad_gap)),
                (lead_length, trace_width_half + pad_gap),
                (0, trace_width_half + pad_gap),
            ]
        )

        # These variables are used to graphically locate the pin locations
        main_pin_line = draw.LineString(
            [(lead_length, trace_width_half), (lead_length, -trace_width_half)]
        )
        
        # Define pin lines along the five main sides
        pin_left = draw.LineString([
            (-(pad_height + taper_height), pad_width_half),
            (-(pad_height + taper_height), -pad_width_half)
        ])

        pin_top = draw.LineString([
            (-(pad_height + taper_height), pad_width_half),
            (-taper_height, pad_width_half)
        ])

        pin_bottom = draw.LineString([
            (-(pad_height + taper_height), -pad_width_half),
            (-taper_height, -pad_width_half)
        ])

        pin_taper_top = draw.LineString([
            (-taper_height, pad_width_half),
            (0, trace_width_half)
        ])

        pin_taper_bottom = draw.LineString([
            (-taper_height, -pad_width_half),
            (0, -trace_width_half)
        ])

        # Create polygon object list
        polys1 = [main_pin_line, transmon, pocket, pin_left, pin_top, pin_bottom, pin_taper_top, pin_taper_bottom]

        # Rotates and translates all the objects as requested. Uses package functions in
        # 'draw_utility' for easy rotation/translation
        polys1 = draw.rotate(polys1, p.orientation, origin=(0, 0))
        polys1 = draw.translate(polys1, xoff=p.pos_x, yoff=p.pos_y)
        [main_pin_line, transmon, pocket, pin_left, pin_top, pin_bottom, pin_taper_top, pin_taper_bottom] = polys1

        # Adds the object to the qgeometry table
        self.add_qgeometry("poly", dict(transmon=transmon), layer=p.layer)
        self.add_qgeometry("poly", dict(pocket=pocket), layer=p.layer, subtract=True)

        self.add_pin(
            "readout",
            points=list(main_pin_line.coords),
            width=trace_width,
            input_as_norm=False
        )

        
        self.add_pin(
            "left",
            points=list(pin_left.coords),
            width=trace_width,
            input_as_norm=False
        )

        self.add_pin(
            "top",
            points=list(pin_top.coords),
            width=trace_width,
            input_as_norm=False
        )

        self.add_pin(
            "bottom",
            points=list(pin_bottom.coords),
            width=trace_width,
            input_as_norm=False
        )

        self.add_pin(
            "taper_top",
            points=list(pin_taper_top.coords),
            width=trace_width,
            input_as_norm=False
        )

        self.add_pin(
            "taper_bottom",
            points=list(pin_taper_bottom.coords),
            width=trace_width,
            input_as_norm=False
        )


