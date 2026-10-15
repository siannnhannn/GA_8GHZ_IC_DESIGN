from qiskit_metal import designs, draw, Dict
from qiskit_metal.qlibrary.core import QComponent
import numpy as np

class GiantAtomWaveguide(QComponent):
    """Giant Atom Waveguide component.

    Inherits 'QComponent' class.

    
    Default Options:
        * trace_width: 'cpw_width' -- Width of the transmission line attached to the transmon
        * trace_gap: 'cpw_gap' -- Gap of the transmission line
        * lead_length: '25um' -- Length of the transmission line attached to the transmon
        * pad_width: '80um' -- Width of the transmon pad
        * pad_height: '80um' -- Height of the transmon pad
        * pad_gap: '58um' -- Gap of the transmon pad
        * taper_height: '122um' -- Height of the taper from the transmon to the transmission line
    """

    default_options = Dict(
            qubit_x="1mm",       # qubit centres at x = -/+ qubit_x
            y_top=".9mm",        # level of the runs into and out of the jogs
            jog_offset="120um",       # jogs at qubit_x -/+ jog_offset
            jog_depth="190um",        # Val's g
            fillet="60um",            # every bend; meander pitch = 2 * fillet
            start_x="1.9mm",          # waveguide ends at x = -/+ start_x
            legs="12",
            legs_top="-0.38mm",       # top of the serpentine's U-turns
            legs_bottom="-1.7248mm",  # bottom of the serpentine's U-turns (solved)
            centre_legs="6",
            centre_x="0.25mm",        # centre meander from x = -centre_x to centre_x
            centre_height="977.9um",  # centre meander height above y_top (solved)
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
            x0, yt, s, g, r = -p.qubit_x, p.y_top, p.jog_offset, p.jog_depth, p.fillet
            pitch = 2 * r
            n = int(p.legs)
            outer_x = p.qubit_x + (n - 1) * pitch / 2   # serpentine centred on the qubit
    # qubit A: in from the left, down its left jog, serpentine, up its right jog
            left = [(-p.start_x, yt), (x0 - s, yt), (x0 - s, yt - g), (-outer_x, yt - g)]
            for k in range(n):
                x = -outer_x + k * pitch
                if k % 2 == 0:   # leg going down
                    if k > 0:
                        left.append((x, p.legs_top))
                    left.append((x, p.legs_bottom))
                else:            # leg going up; the last one up to the jogs' level
                    left.append((x, p.legs_bottom))
                    left.append((x, yt - g if k == n - 1 else p.legs_top))
            left += [(x0 + s, yt - g), (x0 + s, yt)]
    
            # meander between the qubits
            m = int(p.centre_legs)
            centre = []
            for k in range(m):
                x = -p.centre_x + k * 2 * p.centre_x / (m - 1)
                if k % 2 == 0:
                    centre += [(x, yt), (x, yt + p.centre_height)]
                else:
                    centre += [(x, yt + p.centre_height), (x, yt)]
    
            right = [(-x, y) for (x, y) in reversed(left)]
            points = left + centre + right
            self._waveguide_points = np.array(points, dtype=float)
    
           
    
            w = p.trace_width
            self.add_pin("waveguide_connect", input_as_norm=False, width=w,
                         points=np.array([[-p.start_x, yt - w/2], [-p.start_x, yt + w/2]]))
            self.add_pin("waveguide_end", input_as_norm=False, width=w,
                         points=np.array([[p.start_x, yt + w/2], [p.start_x, yt - w/2]]))
    
            
    