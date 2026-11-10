from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent
import numpy as np
from collections import OrderedDict


class CandyCanePurcell(QComponent):

    default_options = Dict(
        pos_x="0um",
        pos_y="0um",

        si_fill="4.6um",

        t_vertical_rect_width="11um",
        t_vertical_rect_length="107um",

        t_horizontal_rect_width="126.7um",
        t_horizontal_rect_length="35.3um",

        rect_centered_width="206.6um",
        rect_centered_length="115.2um",

        purcell_res_separation="10.536um",

        fillet="50um",
        meander_line_width="11um",

        down_from_pin_start="1377um",
        up_from_turn="836.8um",

        up_from_pin_start="1377um",
        down_from_turn="836.8um",

        mini_rect_width="89.282um", # capacitative coupling region
        mini_rect_length="10.9um",
        mini_rect_add_y="750um",
        termination_gap="6um",

        orientation="1",
        rotation="0",

        layer="5",
        chip="main",
    )

    component_metadata = Dict(
        short_name="CandyCanePurcell",
        _qgeometry_table_poly="True",
        _qgeometry_table_path="True",
    )

    def purcell_points(self):
            return self._purcell_points
    
    def resonator_points(self):
        return self._resonator_points

    def rotation_origin(self, p):

        return (
            p.pos_x,
            p.pos_y - p.t_vertical_rect_length / 2,
        )

    def rotate_geometry(self, geometry, p):

        return draw.rotate(
            geometry,
            p.rotation,
            origin=self.rotation_origin(p),
            use_radians=False,
        )

    def rotate_points(self, points, p):

        angle = np.deg2rad(p.rotation)

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])

        origin = np.array(
            self.rotation_origin(p)
        )

        return (
            (points - origin)
            @ rotation_matrix.T
            + origin
        )

    def rounded_centerline(
        self,
        anchors,
        radius,
        points_per_90=24
    ):

        points = [
            np.array(point, dtype=float)
            for point in anchors.values()
        ]

        rounded_points = [
            points[0]
        ]

        for i in range(
            1,
            len(points) - 1
        ):

            p_prev = points[i - 1]
            p_corner = points[i]
            p_next = points[i + 1]

            incoming = (
                p_corner
                - p_prev
            )

            outgoing = (
                p_next
                - p_corner
            )

            incoming_length = np.linalg.norm(
                incoming
            )

            outgoing_length = np.linalg.norm(
                outgoing
            )

            if (
                incoming_length == 0
                or outgoing_length == 0
            ):

                continue

            u_in = (
                incoming
                / incoming_length
            )

            u_out = (
                outgoing
                / outgoing_length
            )

            dot_product = np.dot(
                u_in,
                u_out
            )

            if np.isclose(
                abs(dot_product),
                1.0
            ):

                rounded_points.append(
                    p_corner
                )

                continue

            trim = min(
                radius,
                incoming_length / 2,
                outgoing_length / 2
            )

            tangent_in = (
                p_corner
                - u_in * trim
            )

            tangent_out = (
                p_corner
                + u_out * trim
            )

            cross_product = (
                u_in[0] * u_out[1]
                - u_in[1] * u_out[0]
            )

            if cross_product > 0:

                normal = np.array([
                    -u_in[1],
                    u_in[0]
                ])

            else:

                normal = np.array([
                    u_in[1],
                    -u_in[0]
                ])

            center = (
                tangent_in
                + normal * trim
            )

            start_angle = np.arctan2(
                tangent_in[1]
                - center[1],

                tangent_in[0]
                - center[0]
            )

            end_angle = np.arctan2(
                tangent_out[1]
                - center[1],

                tangent_out[0]
                - center[0]
            )

            if cross_product > 0:

                while (
                    end_angle
                    <= start_angle
                ):

                    end_angle += (
                        2 * np.pi
                    )

            else:

                while (
                    end_angle
                    >= start_angle
                ):

                    end_angle -= (
                        2 * np.pi
                    )

            angles = np.linspace(
                start_angle,
                end_angle,
                points_per_90 + 1
            )

            rounded_points.append(
                tangent_in
            )

            for angle in angles[1:-1]:

                rounded_points.append(
                    center
                    + trim
                    * np.array([
                        np.cos(angle),
                        np.sin(angle)
                    ])
                )

            rounded_points.append(
                tangent_out
            )

        rounded_points.append(
            points[-1]
        )

        return draw.LineString(
            rounded_points
        )

    def make(self):

        p = self.parse_options()

        mirror = (
            p.orientation == -1
        )

        direction = (
            -1 if mirror
            else 1
        )

        t_vertical_x = (
            p.pos_x
        )

        t_vertical_y = (
            p.pos_y
            - p.t_vertical_rect_length / 2
        )

        s = (
            2 * p.mini_rect_width
            + p.purcell_res_separation
        )

        old_x0 = (
            t_vertical_x
            - direction * s
        )

        old_y0 = (
            t_vertical_y
            - p.up_from_pin_start
            - p.t_horizontal_rect_length / 2
            - p.t_vertical_rect_length / 2
        )

        t_vertical = draw.rectangle(
            p.t_vertical_rect_width,
            p.t_vertical_rect_length,
            t_vertical_x,
            t_vertical_y,
        )

        t_horizontal_x = (
            p.pos_x
        )

        t_horizontal_y = (
            p.pos_y
            - p.t_vertical_rect_length
            - p.t_horizontal_rect_length / 2
        )

        t_horizontal = draw.rectangle(
            p.t_horizontal_rect_width,
            p.t_horizontal_rect_length,
            t_horizontal_x,
            t_horizontal_y,
        )

        t_shape = draw.union(
            t_horizontal,
            t_vertical,
        )

        rectangular_pad = draw.rectangle(
            p.rect_centered_width,
            p.rect_centered_length,
            t_horizontal_x,
            t_horizontal_y,
        )

        t_shape_cutout = t_shape.buffer(
            p.si_fill,
            join_style=2,
        )

        large_pad_with_t_gap = (
            rectangular_pad
            .difference(
                t_shape_cutout
            )
            .buffer(0)
        )

        outer_etch = rectangular_pad.buffer(
            p.si_fill,
            join_style=2,
        )

        large_pad_with_t_gap = self.rotate_geometry(
            large_pad_with_t_gap,
            p,
        )

        t_shape = self.rotate_geometry(
            t_shape,
            p,
        )

        outer_etch = self.rotate_geometry(
            outer_etch,
            p,
        )

        t_shape_cutout = self.rotate_geometry(
            t_shape_cutout,
            p,
        )

        self.add_qgeometry(
            "poly",
            {
                "large_pad":
                    large_pad_with_t_gap,

                "t_metal":
                    t_shape,
            },
            layer=p.layer,
            chip=p.chip,
        )

        self.add_qgeometry(
            "poly",
            {
                "big_rect_etch":
                    outer_etch,

                "tiny_t_etch":
                    t_shape_cutout,
            },
            layer=p.layer,
            subtract=True,
            chip=p.chip,
        )

        x_pin = (
            t_horizontal_x
        )

        y_pin = (
            t_horizontal_y
            - p.rect_centered_length / 2
        )

        resonator_start_points = np.array([
            [
                x_pin
                - p.meander_line_width / 2,
                y_pin,
            ],
            [
                x_pin
                + p.meander_line_width / 2,
                y_pin,
            ],
        ])

        resonator_start_points = (
            self.rotate_points(
                resonator_start_points,
                p,
            )
        )

        self.add_pin(
            "resonator_start",
            points=resonator_start_points,
            width=p.meander_line_width,
            input_as_norm=True,
        )

        purcell = self.purcell(
            start_pin_x=old_x0,
            start_pin_y=old_y0,
            p=p,
            direction=direction,
        )

        purcell_min_rect_x = (
            purcell[1]
        )

        purcell_min_rect_y = (
            purcell[2]
        )

        self.resonator(
            start_pin_x=x_pin,
            start_pin_y=y_pin,
            purcell_rect_x=purcell_min_rect_x,
            purcell_rect_y=purcell_min_rect_y,
            p=p,
            direction=direction,
        )

    def resonator(
        self,
        start_pin_x,
        start_pin_y,
        purcell_rect_x,
        purcell_rect_y,
        p,
        direction,
    ):

        x0 = (start_pin_x)

        y0 = (start_pin_y)

        down_from_pin_start = (p.down_from_pin_start)

        up_from_turn = (p.up_from_turn)

        radius_of_turn = (p.fillet)

        diameter_of_turn = (2 * radius_of_turn)

        y_bottom = (
            y0
            - down_from_pin_start
        )

        anchors2 = OrderedDict({

            0: np.array([
                x0,
                y0,
            ]),

            1: np.array([
                x0,
                y_bottom,
            ]),

            2: np.array([
                x0
                + direction
                * diameter_of_turn,

                y_bottom,
            ]),

            3: np.array([
                x0
                + direction
                * diameter_of_turn,

                y_bottom
                + up_from_turn,
            ]),
        })

        resonator_anchor_points = np.array(list(anchors2.values()),dtype=float)
        self._resonator_points = self.rotate_points(resonator_anchor_points,p)

        resonator_centerline = (
            self.rounded_centerline(
                anchors2,
                p.fillet
            )
        )

        resonator_etch = (
            self.rounded_centerline(
                anchors2,
                p.fillet
            )
        )

        resonator_centerline = (
            self.rotate_geometry(
                resonator_centerline,
                p,
            )
        )

        resonator_etch = (
            self.rotate_geometry(
                resonator_etch,
                p,
            )
        )

        self.add_qgeometry(
            "path",
            {
                "resonator":
                    resonator_centerline
            },
            width=p.meander_line_width,
            fillet=0,
            layer=p.layer,
            chip=p.chip,
        )

        rect_x = (
            purcell_rect_x
            + direction
            * (
                p.mini_rect_width / 2
                + p.purcell_res_separation
                + p.mini_rect_width / 2
            )
        )

        rect_y = (
            purcell_rect_y
        )

        rect_resonator = draw.rectangle(
            p.mini_rect_width,
            p.mini_rect_length,
            rect_x,
            rect_y,
        )

        rect_etch = draw.rectangle(
            p.mini_rect_width,
            p.mini_rect_length
            + 2 * p.si_fill,
            rect_x,
            rect_y,
        )

        resonator_tip_x = rect_x - direction * p.mini_rect_width / 2
        resonator_open = draw.rectangle(
            p.termination_gap,
            p.mini_rect_length + 2 * p.si_fill,
            resonator_tip_x - direction * p.termination_gap / 2,
            rect_y,
        )

        rect_resonator = (
            self.rotate_geometry(
                rect_resonator,
                p,
            )
        )

        self._resonator_rect = rect_resonator

        rect_etch = (
            self.rotate_geometry(
                rect_etch,
                p,
            )
        )

        resonator_open = self.rotate_geometry(
            resonator_open,
            p,
        )

        self.add_qgeometry(
            "poly",
            {
                "resonator_mini_rect":
                    rect_resonator
            },
            layer=p.layer,
            chip=p.chip,
        )

        self.add_qgeometry(
            "poly",
            {
                "resonator_mini_rect_etch":
                    rect_etch,
                "resonator_mini_rect_open":
                    resonator_open,
            },
            layer=p.layer,
            subtract=True,
            chip=p.chip,
        )

        self.add_qgeometry(
            "path",
            {
                "resonator_etch":
                    resonator_etch
            },
            width=(
                p.meander_line_width
                + 2 * p.si_fill
            ),
            fillet=0,
            subtract=True,
            layer=p.layer,
            chip=p.chip,
        )

        return anchors2

    def purcell(
        self,
        start_pin_x,
        start_pin_y,
        p,
        direction,
    ):

        x0 = (
            start_pin_x
        )

        y0 = (
            start_pin_y
        )

        up_from_pin_start = (
            p.up_from_pin_start
        )

        down_from_turn = (
            p.down_from_turn
        )

        radius_of_turn = (
            p.fillet
        )

        diameter_of_turn = (
            2 * radius_of_turn
        )

        y_top = (
            y0
            + up_from_pin_start
        )

        anchors = OrderedDict({

            0: np.array([
                x0,
                y0,
            ]),

            1: np.array([
                x0,
                y_top,
            ]),

            2: np.array([
                x0
                - direction
                * diameter_of_turn,

                y_top,
            ]),

            3: np.array([
                x0
                - direction
                * diameter_of_turn,

                y_top
                - down_from_turn,
            ]),
        })

        purcell_anchor_points = np.array(list(anchors.values()),dtype=float)
        self._purcell_points = self.rotate_points(purcell_anchor_points,p)

        pin_points = np.array([
            [
                x0
                + p.meander_line_width / 2,
                y0
            ],
            [
                x0
                - p.meander_line_width / 2,
                y0
            ]
        ])

        pin_points = self.rotate_points(
            pin_points,
            p
        )

        self.add_pin(
            "resonator_connect",
            points=pin_points,
            width=p.meander_line_width,
            input_as_norm=False
        )

        purcell_centerline = (
            self.rounded_centerline(
                anchors,
                p.fillet
            )
        )

        purcell_etch = (
            self.rounded_centerline(
                anchors,
                p.fillet
            )
        )

        purcell_centerline = (
            self.rotate_geometry(
                purcell_centerline,
                p,
            )
        )

        purcell_etch = (
            self.rotate_geometry(
                purcell_etch,
                p,
            )
        )

        self.add_qgeometry(
            "path",
            {
                "purcell":
                    purcell_centerline
            },
            width=p.meander_line_width,
            fillet=0,
            layer=p.layer,
            chip=p.chip,
        )

        self.add_qgeometry(
            "path",
            {
                "purcell_etch":
                    purcell_etch
            },
            width=(
                p.meander_line_width
                + 2 * p.si_fill
            ),
            fillet=0,
            layer=p.layer,
            subtract=True,
            chip=p.chip,
        )

        rect_x = (
            x0
            + direction
            * p.mini_rect_width / 2
        )

        rect_y = (
            y0
            + up_from_pin_start
            - p.mini_rect_add_y
        )

        rect_purcell = draw.rectangle(
            p.mini_rect_width,
            p.mini_rect_length,
            rect_x,
            rect_y,
        )

        rect_etch = draw.rectangle(
            p.mini_rect_width,
            p.mini_rect_length
            + 2 * p.si_fill,
            rect_x,
            rect_y,
        )

        purcell_tip_x = rect_x + direction * p.mini_rect_width / 2
        purcell_open = draw.rectangle(
            p.termination_gap,
            p.mini_rect_length + 2 * p.si_fill,
            purcell_tip_x + direction * p.termination_gap / 2,
            rect_y,
        )

        rect_purcell = (
            self.rotate_geometry(
                rect_purcell,
                p,
            )
        )

        self._purcell_rect = rect_purcell

        rect_etch = (
            self.rotate_geometry(
                rect_etch,
                p,
            )
        )

        purcell_open = self.rotate_geometry(
            purcell_open,
            p,
        )

        self.add_qgeometry(
            "poly",
            {
                "purcell_mini_rect":
                    rect_purcell
            },
            layer=p.layer,
            chip=p.chip,
        )

        self.add_qgeometry(
            "poly",
            {
                "purcell_mini_rect_etch":
                    rect_etch,
                "purcell_mini_rect_open":
                    purcell_open,
            },
            layer=p.layer,
            subtract=True,
            chip=p.chip,
        )

        return (
            anchors,
            rect_x,
            rect_y,
        )
