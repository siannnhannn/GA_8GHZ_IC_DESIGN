import numpy as np

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent

class Squid(QComponent):

    default_options = Dict(
        pos_x = "0um",
        pos_y = "0um",

        t_horizontal_width="2um",
        t_horizontal_length="2um",

        t_vertical_width="2um",
        t_vertical_length="6um",

        side_arm_width="10um",
        side_arm_length="41um",
        side_arm_move_y="0um",

        stem_x="0um",
        stem_width="25um",
        attach_y="0um",

        added_const="0.5um",

        t_layer="5",
        
        squid_opt = Dict(
            plate1_width="3um",
            plate1_height="19um",

            segment_a_length="4um",
            segment_a_width="3um",

            JJ_gap="2um",

            segment_c_width="4um",
            segment_c_height="20um",

            segment_d_length="7.5um",
            segment_d_width="4um",

            plate2_width="7um",
            plate2_height="12um",

            extension_width="7um", #extension is the extra rectangle that attaches to the lower part of plate 1 that covers the T in the Si layer
            extension_length="10um",
    
    
            sides_lengths=("4um", "2.07571198628um", "2.07571198628um"), #this calls on my function to make the JJ and it takes base, and the two sides as inputs (of a triangle)

            #two rectangles are specified cause there are two different jjs with a different width on either side of squid
            rectangle1_width="0.4um",
            rectangle1_length="1.36um",
    
            rectangle2_width="0.17um",
            # Compensate the narrower finger's later triangle contact point so
            # its Dolan opening matches junction 1 at exactly 140 nm.
            rectangle2_length="1.32805875um",

            rotation=0,
    
            layer_60_length_1="1.5um",
            layer_60_length_2="1.46805875um", #the jjs live on a different layer than the squid.
        ),

        transmon_side = "1",

        layer = "20"
    )
    
    component_metadata = Dict(
        short_name="SQUID",
        _qgeometry_table_poly="True"
    )

    def make(self):
        p = self.parse_options()

        pin_center = self.make_connection_arm(p, p.stem_x, p.attach_y, p.stem_width)
        
        self.extension_plate_center = self.make_squid(p, p.squid_opt, pin_center[0], pin_center[1])

        self.make_t_si(p, self.extension_plate_center[0], self.extension_plate_center[1])


        
    def make_squid(self, p, squid_opt, attach_x, attach_y):
        s = squid_opt
        #PLATE 1:
        #init pos of squid is determined by plate 1
        x_1 = 0
        y_1 = 0
        mirror = (p.transmon_side == -1)

        plate1 = draw.rectangle(s.plate1_width, s.plate1_height, x_1, y_1)

        #SEGMENT A: 
        x_a_upper = x_1 + (s.plate1_width/2) + (s.segment_a_width/2)
        y_a_upper = y_1 + (s.plate1_height/2) - (s.segment_a_length/2)
        segment_a = draw.rectangle(
                    s.segment_a_width,
                    s.segment_a_length,
                    x_a_upper,
                    y_a_upper
                        )

        #SEGMENT C:
        x_c = x_a_upper + (s.segment_a_width/2) + s.JJ_gap + (s.segment_c_width/2)
        y_c = y_a_upper + (s.segment_a_length/2) - (s.segment_c_height/2)
        
        segment_c = draw.rectangle(
            s.segment_c_width,
            s.segment_c_height,
            x_c,
            y_c
        )

        #SEGMENT A LOWER:
        y_a_lower = y_c - (s.segment_c_height/2) + (s.segment_a_length/2)
        segment_a_lower = draw.rectangle(
            s.segment_a_width,
            s.segment_a_length,
            x_a_upper,
            y_a_lower
        )

        #SEGMENT D: 
        x_d = x_c + (s.segment_c_width/2) + (s.segment_d_length/2) -0.001
        segment_d = draw.rectangle(
            s.segment_d_length,
            s.segment_d_width,
            x_d,
            y_c
        )

        #PLATE 2:
        x_plate2 = x_d + (s.segment_d_length/2) + (s.plate2_width/2)
                
        plate2 = draw.rectangle(
            s.plate2_width,
            s.plate2_height,
            x_plate2,
            y_c
        )

        #EXTENSION PLATE:
        x_extension_plate = x_1 + (s.plate1_width/2) - (s.extension_width/2)
        y_extension_plate = y_1 - (s.plate1_height/2) - (s.extension_length/2)
        extension_plate = draw.rectangle(
            s.extension_width,
            s.extension_length,
            x_extension_plate,
            y_extension_plate
        )

        #Make the squid with all of the components, no JJ yet
        squid_no_jj = draw.union(
                    plate1,
                    segment_a,
                    segment_a_lower,
                    extension_plate,
                    segment_c,
                    segment_d,
                    plate2
                )

        #Now we proceed to make the two JJs.
        #JJ1
        jj1 = self.jj_squid(
                    s.sides_lengths,
                    s.rectangle1_width,
                    s.rectangle1_length
                )

        jj1_layer60 = self.jj_squid(
                    s.sides_lengths,
                    s.rectangle1_width,
                    s.layer_60_length_1
                ) #the much longer jj1 that lives in a different layer
        
        jj_1 = draw.rotate(jj1, 270, origin=(0, 0))
        jj_1_layer60 = draw.rotate(jj1_layer60, 270, origin=(0, 0))

        #JJ2
        jj2 = self.jj_squid(
                    s.sides_lengths,
                    s.rectangle2_width,
                    s.rectangle2_length
                )
        jj2_layer60 = self.jj_squid(
                            s.sides_lengths,
                            s.rectangle2_width,
                            s.layer_60_length_2
                        )
        jj_2 = draw.rotate(jj2, 270, origin=(0, 0))
        jj_2_layer60 = draw.rotate(jj2_layer60, 270, origin=(0, 0))
        x_jj = x_a_upper + s.segment_a_width / 2

        jj_1 = draw.translate(
            jj_1,
            xoff=x_jj,
            yoff=y_a_lower
        )

        jj_1_60 = draw.translate(
            jj_1_layer60,
            xoff=x_jj,
            yoff=y_a_lower
        )

        jj_2 = draw.translate(
            jj_2,
            xoff=x_jj,
            yoff=y_a_upper
        )

        jj_2_60 = draw.translate(
            jj_2_layer60,
            xoff=x_jj,
            yoff=y_a_upper
        )

        if mirror:
            # Mirror the SQUID body and the JJ *positions*, but preserve the
            # global +x direction of every JJ shape for foundry consistency.
            squid_no_jj = draw.scale(
                squid_no_jj, xfact=-1, yfact=1, origin=(0, 0)
            )
            # Place the +x-facing pair across the mirrored physical JJ gap,
            # using one shared base offset for layer 20 and layer 60.
            mirrored_base_x = -(x_c - s.segment_c_width / 2)
            junction_xoff = mirrored_base_x - x_jj
            jj_1 = draw.translate(jj_1, xoff=junction_xoff)
            jj_2 = draw.translate(jj_2, xoff=junction_xoff)
            jj_1_60 = draw.translate(jj_1_60, xoff=junction_xoff)
            jj_2_60 = draw.translate(jj_2_60, xoff=junction_xoff)

        design1 = draw.union(squid_no_jj, jj_1, jj_2)
        design_layer60 = draw.union(jj_1_60, jj_2_60)

        #this plate2_anchor will allow us to connect it to the T on the transmon.
        plate2_anchor = draw.Point(
                            x_plate2,
                            y_c
                        )

        if mirror:
            plate2_anchor = draw.scale(
                plate2_anchor,
                xfact=-1,
                yfact=1,
                origin=(0, 0)
            )

        #we rotate it as we are going to the same with the entire design based off user input
        plate2_anchor_rotated = draw.rotate(
                    plate2_anchor,
                    s.rotation,
                    origin=(0, 0)
                )
        x_translation = attach_x - plate2_anchor_rotated.x
        y_translation = attach_y - plate2_anchor_rotated.y

        #This extension plate anchor will act as the coordinates of the second T in the Si:
        extension_plate_anchor = draw.Point(
                            x_extension_plate,
                            y_extension_plate
                        )
        if mirror:
            extension_plate_anchor = draw.scale(
                extension_plate_anchor,
                xfact=-1,
                yfact=1,
                origin=(0, 0)
                    )
            
        extension_plate_anchor_rotated = draw.rotate(
                            extension_plate_anchor,
                            s.rotation,
                            origin=(0, 0)
                        )

        x_coordinate_extension_plate = (
            attach_x
            + extension_plate_anchor_rotated.x
            - plate2_anchor_rotated.x
            )
        
        y_coordinate_extension_plate = (
            attach_y
            + extension_plate_anchor_rotated.y
            - plate2_anchor_rotated.y
        )

        #we rotate the squid if needed
        design1 = draw.rotate(
                    design1,
                    s.rotation,
                    origin=(0, 0)
                )

        #rename the center of  the design based on the location of the pin on plate 2.
        design1 = draw.translate(
                    design1,
                    xoff=x_translation,
                    yoff=y_translation
                )

        #This also includes a rotation of the JJS:
        design_layer60 = draw.rotate(
                    design_layer60,
                    s.rotation,
                    origin=(0, 0)
                )
        design_layer60 = draw.translate(
                    design_layer60,
                    xoff=x_translation,
                    yoff=y_translation
                )
        # SE1JJ is a device-definition layer: the foundry requires every
        # completed polygon vertex on the 5 nm manufacturing grid.
        design_layer60 = draw.shapely.set_precision(design_layer60, 0.000005)
        
        geom = {"design": design1}

    

        self.add_qgeometry("poly", geom, layer=int(p.layer), subtract=False)
        self.add_qgeometry(
            "poly",
            {"jj_layer60": design_layer60},
            layer=60,
            subtract=False
        )
        return x_coordinate_extension_plate, y_coordinate_extension_plate

    def jj_squid(self, sides_lengths, rectangle_width, rectangle_length):
        c = sides_lengths[0]  # base of the triangle
        a = sides_lengths[1]  # side from B to C
        b = sides_lengths[2]  # side from A to C

        # Triangle vertices
        vertex_A = (-c / 2, 0)
        vertex_B = ( c / 2, 0)
        vertex_C_x =  0
        vertex_C_y = np.sqrt(b**2 - (c / 2)**2)

        triangle = draw.Polygon([
            vertex_A,
            vertex_B,
            (vertex_C_x, vertex_C_y)
        ])


        if rectangle_width <= 0:
            raise ValueError("rectangle_width must be positive.")

        if rectangle_width > c:
            raise ValueError(
                "rectangle_width cannot be greater than the triangle base."
            )

        
        width_fraction = rectangle_width / c

        overlap_depth = vertex_C_y * width_fraction

        rectangle_bottom_y = vertex_C_y - overlap_depth

        left_contact_x = (
            vertex_C_x
            + width_fraction * (vertex_A[0] - vertex_C_x)
        )

        right_contact_x = (
            vertex_C_x
            + width_fraction * (vertex_B[0] - vertex_C_x)
        )
        rectangle_center_x = (left_contact_x + right_contact_x) / 2

        rectangle_center_y = rectangle_bottom_y + rectangle_length / 2

        rectangle = draw.rectangle(
            rectangle_width,
            rectangle_length,
            rectangle_center_x,
            rectangle_center_y
        )

        jj = draw.union(triangle, rectangle)

        return jj
    
    def make_t_si(self, p, center_x, center_y):

        t_si_horizontal_local = draw.rectangle(p.t_horizontal_length+p.added_const, p.t_horizontal_width, 0, 0)
        t_si_vertical_local = draw.rectangle(
            p.t_vertical_width,
            p.t_vertical_length,
            -p.t_horizontal_length / 2
            - p.t_vertical_width / 2,
            0
        )

        t_cutout_shape_si_local = draw.union(t_si_horizontal_local,t_si_vertical_local).buffer(0)
        
        if p.transmon_side == -1:
            t_cutout_shape_si_local = draw.scale(
                t_cutout_shape_si_local,
                xfact=-1,
                yfact=1,
                origin=(0, 0)
            )

        local_t_centroid = t_cutout_shape_si_local.centroid
        
        t_cutout_shape_si = draw.translate(
            t_cutout_shape_si_local,
            xoff=center_x - local_t_centroid.x,
            yoff=center_y - local_t_centroid.y
        )

        self.add_qgeometry(
            "poly",
            {"t_si": t_cutout_shape_si},
            layer=int(p.t_layer),
            subtract=True
        )
    def side_arm_placement_and_cutout(self, side_arm, mini_t_horizontal_rectangle, mini_t_vertical_rectangle, p, stem_x, attach_y, stem_width):
            "specifies the two transmon_side of the side arm"
            y_side_arm = attach_y + p.side_arm_move_y
            if p.transmon_side == 1:
                x_side_arm = stem_x - stem_width/2 - p.side_arm_length/2 + 0.00033
                x_t_horizontal = x_side_arm - p.side_arm_length/2 + p.t_horizontal_length/2
                x_t_vertical = x_t_horizontal + p.t_horizontal_length/2 + p.t_vertical_width/2
                
            elif p.transmon_side == -1:
                x_side_arm = stem_x + stem_width/2 + p.side_arm_length/2 - 0.00033
                x_t_horizontal = x_side_arm + p.side_arm_length/2 - p.t_horizontal_length/2
                x_t_vertical = x_t_horizontal - p.t_horizontal_length/2 - p.t_vertical_width/2
            else:
                raise ValueError("transmon_side must be 1 or -1")
    
            return [draw.translate(side_arm, x_side_arm, y_side_arm), draw.translate(mini_t_horizontal_rectangle, x_t_horizontal, y_side_arm), 
                    draw.translate(mini_t_vertical_rectangle, x_t_vertical, y_side_arm)]

    def make_connection_arm(self, p, stem_x, attach_y, stem_width):

        print("=== ARM DEBUG ===")

        for key in [
            "side_arm_length",
            "side_arm_width",
            "side_arm_move_y",
            "t_horizontal_length",
            "t_horizontal_width",
            "t_vertical_length",
            "t_vertical_width",
            "stem_x",
            "stem_width",
            "attach_y"
        ]:
            value = p.get(key, "MISSING")
            print(key, "=", repr(value), type(value))

            
        side_arm = draw.rectangle(p.side_arm_length, p.side_arm_width, 0, 0)
        mini_t_horizontal_rectangle = draw.rectangle(p.t_horizontal_length, p.t_horizontal_width, 0, 0)
        mini_t_vertical_rectangle = draw.rectangle(p.t_vertical_width, p.t_vertical_length, 0, 0)

        arm, t_horizontal, t_vertical = (
            self.side_arm_placement_and_cutout(
                side_arm,
                mini_t_horizontal_rectangle,
                mini_t_vertical_rectangle,
                p,
                stem_x,
                attach_y,
                stem_width
            )
        )
        t_cutout = draw.union(t_horizontal, t_vertical)

        t_center_x = t_cutout.centroid.x
        t_center_y = t_cutout.centroid.y
        
        pin_half_length = p.t_vertical_width / 2

        squid_pin_points = np.array([
            [t_center_x - pin_half_length, t_center_y],
            [t_center_x + pin_half_length, t_center_y]
        ])

        self.add_pin(
            "squid_center",
            points=squid_pin_points,
            width=p.t_vertical_width
        )

        arm_with_cutout = arm.difference(t_cutout).buffer(0)

        self.add_qgeometry(
            "poly",
            {"connection_arm": arm_with_cutout},
            layer=int(p.t_layer),
            subtract=False
        )

        if p.transmon_side == 1:
            stem_edge_x = stem_x - stem_width / 2
        else:
            stem_edge_x = stem_x + stem_width / 2

        arm_y = attach_y + p.side_arm_move_y

        pin_points = np.array([
            [stem_edge_x, arm_y - p.side_arm_width / 2],
            [stem_edge_x, arm_y + p.side_arm_width / 2]
        ])

        self.add_pin(
            "stem_attach",
            points=pin_points,
            width=p.side_arm_width
        )
        return self.pins["squid_center"]["middle"]

        