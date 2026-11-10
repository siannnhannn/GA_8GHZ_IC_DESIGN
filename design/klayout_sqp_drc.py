"""Lightweight SQUILL BISQuit preflight DRC for KLayout 0.30+.

This is a transparent, best-effort implementation of the public SQP 3.0 rule
tables.  It is not a replacement for the encrypted MIT LL Calibre deck.

Batch usage:
    klayout -b -zz -r klayout_sqp_drc.py \
      -rd input=design.gds -rd report=design_sqp_preflight.lyrdb

Interactive usage (opens the GDS and the Marker Browser):
    klayout design.gds -rr klayout_sqp_drc.py -rd report=report.lyrdb

Optional run-time definitions:
    -rd top=CELL_NAME       choose a top cell when the GDS has more than one
    -rd dfm=true            enable advisory DFM thresholds where implemented
    -rd local=true          enable ALMA project checks (45-degree BR angles)

With no ``input`` definition, the script checks the active KLayout cellview.
The marker database can be opened in KLayout with Tools > Marker Browser.
"""

from __future__ import annotations

import math
from pathlib import Path

import pya


VERSION = "1.2"
EXPECTED_DBU_UM = 0.0005  # PRECISION 2000: 2000 database units per micron.
EPS_UM = 0.001

LAYERS = {
    "POST": (1, 0), "POSTCLEAN": (3, 0),
    "M1": (5, 0), "M1FILL": (5, 2), "M1PAD": (5, 11),
    "M2": (10, 0), "M2FILL": (10, 2), "M2PAD": (10, 11),
    "SE1": (20, 0), "RR": (30, 0), "BR": (31, 0),
    "UBM": (40, 0), "BUMP": (45, 0), "SEB1": (55, 0),
    "SE1_DEVICE": (60, 0),
    "IPOST": (101, 0), "IPOSTCLEAN": (103, 0),
    "IM1": (105, 0), "IM1FILL": (105, 2), "IM1PAD": (105, 11),
    "IM2": (110, 0), "IM2FILL": (110, 2), "IM2PAD": (110, 11),
    "ISE1": (120, 0), "IRR": (130, 0), "IBR": (131, 0),
    "IUBM": (140, 0), "IBUMP": (145, 0), "ISEB1": (155, 0),
    "ISE1_DEVICE": (160, 0), "CHIP_EDGE": (703, 0),
    "ICHIP_EDGE": (713, 0),
}


def _runtime_bool(name: str, default: bool) -> bool:
    value = globals().get(name, str(default))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SQPPreflight:
    def __init__(self, layout: pya.Layout, cell: pya.Cell, report_path: Path,
                 dfm: bool = False, local_checks: bool = True):
        self.layout = layout
        self.cell = cell
        self.dbu = layout.dbu
        self.report_path = report_path
        self.dfm = dfm
        self.local_checks = local_checks
        self.rdb = pya.ReportDatabase("SQUILL SQP 3.0 KLayout preflight")
        self.rdb.description = (
            "Transparent preflight based on the SQP 3.0 public rule tables; "
            "official signoff still requires the supplied encrypted Calibre deck."
        )
        self.rdb.generator = f"klayout_sqp_drc.py {VERSION}"
        self.rdb.original_file = str(globals().get("input", "active cellview"))
        self.rdb_cell = self.rdb.create_cell(cell.name)
        self.categories = {}
        self.counts = {}
        self.notes = []

    def u(self, microns: float) -> int:
        return int(round(microns / self.dbu))

    def region(self, name: str) -> pya.Region:
        layer, datatype = LAYERS[name]
        index = self.layout.layer(layer, datatype)
        return pya.Region(self.cell.begin_shapes_rec(index)).merged()

    def union(self, *names: str) -> pya.Region:
        result = pya.Region()
        for name in names:
            result |= self.region(name)
        return result.merged()

    def base_region(self, name: str) -> pya.Region:
        fill = {"M1": "M1FILL", "M2": "M2FILL",
                "IM1": "IM1FILL", "IM2": "IM2FILL"}[name]
        return self.union(name, fill)

    def category(self, rule: str, description: str):
        if rule not in self.categories:
            self.categories[rule] = self.rdb.create_category(
                None, f"{rule}: {description}"
            )
            self.counts[rule] = 0
        return self.categories[rule]

    def add(self, rule: str, description: str, shape) -> None:
        category = self.category(rule, description)
        # RDB values use micron-space geometry.  Edge pairs are widened slightly
        # so coincident/point checks remain visible in the Marker Browser.
        if isinstance(shape, pya.EdgePair):
            shape = shape.normalized().polygon(max(1, self.u(0.002)))
        item = self.rdb.create_item(self.rdb_cell.rdb_id(), category.rdb_id())
        item.add_value(shape.to_dtype(self.dbu))
        self.counts[rule] += 1

    def add_region(self, rule: str, description: str, region: pya.Region) -> None:
        for polygon in region.each():
            self.add(rule, description, polygon)

    def add_pairs(self, rule: str, description: str, pairs: pya.EdgePairs) -> None:
        for pair in pairs.each():
            self.add(rule, description, pair)

    def add_box(self, rule: str, description: str, box: pya.Box) -> None:
        self.add(rule, description, box)

    def check_rectangle(self, rule: str, description: str,
                        region: pya.Region, exact_size=None) -> None:
        for polygon in region.each_merged():
            bad = not polygon.is_box()
            if exact_size is not None:
                bbox = polygon.bbox()
                dims = sorted((bbox.width(), bbox.height()))
                target = sorted(self.u(x) for x in exact_size)
                bad = bad or dims != target
            if bad:
                self.add(rule, description, polygon)

    def rectangle_sides_um(self, polygon: pya.Polygon):
        """Return side lengths for any rotated rectangle, or None."""
        points = list(polygon.each_point_hull())
        if len(points) != 4:
            return None
        vectors = [(points[(i + 1) % 4].x - points[i].x,
                    points[(i + 1) % 4].y - points[i].y) for i in range(4)]
        scale = max(dx * dx + dy * dy for dx, dy in vectors)
        tolerance = max(1, scale) * 1e-6
        if any(abs(vectors[i][0] * vectors[(i + 1) % 4][0] +
                   vectors[i][1] * vectors[(i + 1) % 4][1]) > tolerance
               for i in range(4)):
            return None
        lengths = sorted(math.hypot(dx, dy) * self.dbu for dx, dy in vectors)
        if abs(lengths[0] - lengths[1]) > EPS_UM or abs(lengths[2] - lengths[3]) > EPS_UM:
            return None
        return lengths[0], lengths[2]

    def check_min_area(self, rule: str, description: str,
                       region: pya.Region, area_um2: float) -> None:
        threshold = area_um2 / (self.dbu * self.dbu)
        for polygon in region.each_merged():
            if polygon.area() + 0.5 < threshold:
                self.add(rule, description, polygon)

    def check_hole_area(self, rule: str, description: str,
                        region: pya.Region, area_um2: float) -> None:
        threshold = area_um2 / (self.dbu * self.dbu)
        for polygon in region.each_merged():
            for hole_index in range(polygon.holes()):
                hole = pya.Polygon(list(polygon.each_point_hole(hole_index)))
                if hole.area() + 0.5 < threshold:
                    self.add(rule, description, hole)

    def check_edge(self, edge_name: str, edge: pya.Region) -> None:
        if edge.is_empty():
            return
        desc = "ChipEdge must be one orthogonal rectangle centered at (0,0)"
        polygons = list(edge.each_merged())
        if len(polygons) != 1 or not polygons[0].is_box():
            self.add_region(f"{edge_name}.1", desc, edge)
            return
        bbox = polygons[0].bbox()
        if bbox.left + bbox.right != 0 or bbox.bottom + bbox.top != 0:
            self.add_box(f"{edge_name}.1", desc, bbox)

        physical = self.union(
            "M1", "M1FILL", "M2", "M2FILL", "SE1", "SEB1", "RR", "BR",
            "UBM", "BUMP", "IM1", "IM1FILL", "IM2", "IM2FILL", "ISE1",
            "ISEB1", "IRR", "IBR", "IUBM", "IBUMP"
        )
        reserved = pya.Region(pya.Box(
            bbox.left, bbox.top - self.u(100),
            bbox.left + self.u(300), bbox.top,
        ))
        self.add_region(f"{edge_name}.2", "Upper-left 300 x 100 um corner must not contain physical metal",
                        physical & reserved)

    def check_base_metal(self, name: str, edge: pya.Region) -> None:
        metal = self.base_region(name)
        prefix = name
        width = 3.0 if self.dfm else 2.0
        area = 16.0 if self.dfm else 12.5
        suffix = ".DFM" if self.dfm else ""
        self.add_pairs(f"{prefix}.W.1{suffix}", f"{name} minimum width {width} um",
                       metal.width_check(
                           self.u(width), False, pya.Region.Euclidian,
                           45, None, None,
                       ))
        self.add_pairs(f"{prefix}.S.1{suffix}", f"{name} minimum space {width} um",
                       metal.space_check(
                           self.u(width), False, pya.Region.Euclidian,
                           80, None, None,
                       ))
        self.check_min_area("MET.A.1" + suffix, f"Base-metal minimum area {area} um^2", metal, area)
        self.check_hole_area("MET.A.2" + suffix, f"Base-metal minimum enclosed-hole area {area} um^2", metal, area)
        self.add_region("MET.CE.E.1", f"{name} must be inside its ChipEdge", metal - edge)

    def check_posts(self, post_name: str, clean_name: str, same_metal: pya.Region,
                    opposite_metal: pya.Region, same_se: pya.Region,
                    opposite_se: pya.Region, same_seb: pya.Region,
                    same_br: pya.Region, opposite_br: pya.Region,
                    same_ubm: pya.Region, opposite_ubm: pya.Region,
                    edge: pya.Region, opposite_edge: pya.Region) -> None:
        post = self.region(post_name)
        if post.is_empty():
            return
        self.check_rectangle("POST.1", "Post must be rectangular", post, (100.0, 100.0))
        checks = (
            (same_metal, 25, "POST.MET.S.1"), (same_se, 350, "POST.SE.S.1"),
            (opposite_se, 50, "POST.SE.S.2"),
            (same_seb, 350 if self.dfm else 50, "POST.SEB.S.1"),
            (same_br, 75, "POST.BR.S.1"), (opposite_br, 25, "POST.BR.S.2"),
            (same_ubm, 35, "POST.UBM.S.1"), (opposite_ubm, 5, "POST.UBM.S.2"),
        )
        for other, distance, rule in checks:
            if not other.is_empty():
                self.add_pairs(rule, f"Post separation must be at least {distance} um",
                               post.separation_check(other, self.u(distance)))
        self.add_region("POST.MET.E.1", "Post needs 10 um opposite-tier base-metal enclosure",
                        post - opposite_metal.sized(-self.u(10)))
        both_safe = edge.sized(-self.u(200)) & opposite_edge.sized(-self.u(200))
        if not both_safe.is_empty():
            self.add_region("POST.CE.E.1", "Post needs 200 um enclosure by both chip edges",
                            post - both_safe)
        clean = self.region(clean_name)
        self.add_region("POST.PC.E.1", "PostClean must enclose Post by exactly 20 um",
                        post.sized(self.u(20)) ^ clean)

    def check_se(self, se_name: str, seb_name: str, device_name: str,
                 metal: pya.Region, edge: pya.Region) -> None:
        se = self.region(se_name)
        seb = self.region(seb_name)
        device = self.region(device_name)
        prefix = "I" if se_name.startswith("I") else ""
        if not device.is_empty():
            self.add_pairs(prefix + "SE.DEV.1", "SE_device vertices must be on a 5 nm grid",
                           device.grid_check(self.u(0.005), self.u(0.005)))
            gaps = (device - se).merged()
            for polygon in gaps.each_merged():
                bbox = polygon.bbox()
                if (not polygon.is_box() or
                        abs(bbox.width() * self.dbu - 0.140) > self.dbu / 2):
                    self.add(prefix + "SE.DEV.2", "Dolan gap must be an orthogonal rectangle exactly 0.140 um wide in x", polygon)
            if not seb.is_empty():
                self.add_pairs(prefix + "SE.DEV.SEB.S.1", "SE_device to SEB space >= 2 um",
                               device.separation_check(seb, self.u(2.0)))
            self.add_pairs(prefix + "SE.DEV.MET.S.1", "SE_device to same-tier base metal space >= 4 um",
                           device.separation_check(metal, self.u(4.0)))

        # Exclude a 2 nm halo around the device-definition polygons.  SE1JJ
        # vertices are snapped to a 5 nm grid, while the overlapping SE1
        # construction may land on the 0.5 nm GDS grid; without this tolerance
        # their harmless registration residue is misread as sub-0.5 um routing.
        routing_fine = (se - device.sized(self.u(0.002))).merged()
        routing_broad = (seb - device).merged()
        if not routing_fine.is_empty():
            self.add_pairs(prefix + "SE.W.1", "SE fine routing minimum width 0.5 um",
                           routing_fine.width_check(self.u(0.5)))
        if not routing_broad.is_empty():
            self.add_pairs(prefix + "SEB.W.1", "SE broad routing minimum width 0.5 um",
                           routing_broad.width_check(self.u(0.5)))
        combined = (se | seb | device).merged()
        if not combined.is_empty():
            self.add_pairs(prefix + "SE.S.1", "SE minimum space 2 um outside device regions",
                           combined.isolated_check(self.u(2.0)))
            # Notches belonging to an SE1JJ device are intentional Dolan
            # geometry and explicitly outside SE.S.1's scope.  Keep genuine
            # routing notches, but do not duplicate them through space_check.
            for pair in combined.notch_check(self.u(2.0)).each():
                marker = pya.Region(pair.normalized().polygon(max(1, self.u(0.002))))
                if (marker & device).is_empty():
                    self.add(prefix + "SE.S.1",
                             "SE minimum notch 2 um outside device regions",
                             pair)
            self.add_region(prefix + "SE.CE.E.1", "SE must be enclosed by ChipEdge by 200 um",
                            (se | seb) - edge.sized(-self.u(200)))

        overlap = ((se | seb) & metal).merged()
        self.check_min_area(prefix + "SE.MET.O.1", "Each SE/base-metal overlap area >= 44 um^2",
                            overlap, 44.0)
        density = se.area() * self.dbu * self.dbu / 10000.0
        density += seb.area() * self.dbu * self.dbu / 865000.0
        if density > 1.0 and not edge.is_empty():
            self.add_box(prefix + "SE.D.1", f"Advisory write-time density is {density:.4f} (> 1.0)", edge.bbox())

    @staticmethod
    def _edge_angle_ok(p1, p2, tolerance_deg=0.05) -> bool:
        """Accept nominal 45-degree edges after GDS integer-grid rounding.

        A 45-degree rotation of the foundry bridge dimensions generally lands
        between 0.5 nm database-grid points.  The resulting polygon edges can
        differ from 45 degrees by about 0.014 degrees even though the component
        orientation is exactly 45 degrees.  The tolerance remains far below a
        visually or lithographically meaningful arbitrary rotation.
        """
        dx, dy = p2.x - p1.x, p2.y - p1.y
        if dx == 0 and dy == 0:
            return True
        angle = math.degrees(math.atan2(dy, dx)) % 45.0
        error = min(angle, 45.0 - angle)
        return error <= tolerance_deg

    def check_bridges(self, br_name: str, rr_name: str, other_br_name: str,
                      landing: pya.Region, edge: pya.Region) -> None:
        br = self.region(br_name)
        rr = self.region(rr_name)
        prefix = "I" if br_name.startswith("I") else ""
        for polygon in br.each_merged():
            if polygon.num_points_hull() != 12:
                self.add(prefix + "BR.2", "Merged BR dogbone must have exactly 12 vertices", polygon)
            points = list(polygon.each_point_hull())
            if self.local_checks and any(
                    not self._edge_angle_ok(points[i], points[(i + 1) % len(points)])
                    for i in range(len(points))):
                self.add("LOCAL.BR.ANGLE", "Airbridge edges must be at 0, 45, 90 or 135 degrees", polygon)
            center2 = pya.Point(polygon.bbox().left + polygon.bbox().right,
                                polygon.bbox().bottom + polygon.bbox().top)
            reflected = [(center2.x - p.x, center2.y - p.y) for p in points]
            symmetry_tolerance = self.u(0.002)  # 2 nm GDS rotation/rounding allowance
            symmetric = all(
                min((rx - p.x) ** 2 + (ry - p.y) ** 2 for p in points)
                <= symmetry_tolerance ** 2
                for rx, ry in reflected
            )
            if not symmetric:
                self.add(prefix + "BR.1", "Bridge must be centered and 180-degree symmetric", polygon)
            if self.local_checks:
                local_rr = rr.interacting(pya.Region(polygon))
                tethers = list(local_rr.each_merged())
                if len(tethers) == 2:
                    rr_center2_x = sum(t.bbox().left + t.bbox().right for t in tethers) / 2
                    rr_center2_y = sum(t.bbox().bottom + t.bbox().top for t in tethers) / 2
                    br_center2_x = polygon.bbox().left + polygon.bbox().right
                    br_center2_y = polygon.bbox().bottom + polygon.bbox().top
                    if (abs(rr_center2_x - br_center2_x) > self.u(0.002) or
                            abs(rr_center2_y - br_center2_y) > self.u(0.002)):
                        self.add("LOCAL.BR.CENTER", "Airbridge must be centered between its two RR tethers", polygon)

        if not br.is_empty():
            self.add_pairs(prefix + "BR.S.1", "BR minimum space 5 um",
                           br.isolated_check(self.u(5.0)))
            other = self.region(other_br_name)
            if not other.is_empty():
                self.add_pairs(prefix + "BR.IBR.S.1", "Opposite-tier bridge space >= 10 um",
                               br.separation_check(other, self.u(10.0)))
            self.add_region(prefix + "BR.CE.E.1", "BR must be enclosed by ChipEdge by 200 um",
                            br - edge.sized(-self.u(200)))
        if not rr.is_empty():
            self.add_region(prefix + "RR.E.1", "RR must have 1.5 um enclosure by base metal or SE",
                            rr - landing.sized(-self.u(1.5)))
            for polygon in rr.each_merged():
                dims = self.rectangle_sides_um(polygon)
                valid = dims is not None and any(
                    abs(dims[0] - short) <= 0.003 and abs(dims[1] - long) <= 0.003
                    for short, long in ((8.0, 8.0), (10.0, 10.5), (13.0, 14.0))
                )
                if not valid:
                    self.add(prefix + "RR.WL.2", "RR must be a rotated rectangle sized 8x8, 10x10.5, or 13x14 um", polygon)

    def check_ubm_bumps(self, ubm_name: str, bump_name: str, metal: pya.Region,
                        edge: pya.Region) -> None:
        ubm, bump = self.region(ubm_name), self.region(bump_name)
        prefix = "I" if ubm_name.startswith("I") else ""
        if not ubm.is_empty():
            self.add_pairs(prefix + "UBM.W.1", "UBM minimum width 3 um", ubm.width_check(self.u(3)))
            self.add_pairs(prefix + "UBM.S.1", "UBM minimum space 3 um", ubm.space_check(self.u(3)))
            self.add_pairs(prefix + "UBM.S.1", "UBM minimum notch 3 um", ubm.notch_check(self.u(3)))
            self.check_min_area(prefix + "UBM.A.1", "UBM minimum area 25 um^2", ubm, 25)
            self.check_hole_area(prefix + "UBM.A.2", "UBM minimum hole area 25 um^2", ubm, 25)
            self.add_region(prefix + "UBM.MET.E.1", "UBM enclosed by base metal by 5 um",
                            ubm - metal.sized(-self.u(5)))
        if bump.is_empty():
            return
        for polygon in bump.each_merged():
            bbox = polygon.bbox()
            diameter_ok = (abs(bbox.width() * self.dbu - 15) <= EPS_UM and
                           abs(bbox.height() * self.dbu - 15) <= EPS_UM)
            if polygon.num_points_hull() < 40 or polygon.num_points_hull() > 50 or not diameter_ok:
                self.add(prefix + "BUMP.1", "Bump must be a 15 um circle with 40-50 vertices", polygon)
        self.add_pairs(prefix + "BUMP.S.1", "Bump minimum edge-to-edge space 20 um",
                       bump.space_check(self.u(20)))
        self.add_region(prefix + "BUMP.UBM.E.1", "Bump enclosed by UBM by 7.5 um",
                        bump - ubm.sized(-self.u(7.5)))
        if not edge.is_empty():
            b = edge.bbox()
            s = self.u(200)
            corners = pya.Region()
            for box in (pya.Box(b.left, b.bottom, b.left+s, b.bottom+s),
                        pya.Box(b.right-s, b.bottom, b.right, b.bottom+s),
                        pya.Box(b.left, b.top-s, b.left+s, b.top),
                        pya.Box(b.right-s, b.top-s, b.right, b.top)):
                corners.insert(box)
            self.add_region(prefix + "BUMP.3", "No bumps in 200 x 200 um ChipEdge corners",
                            bump & corners)

    def check_pads(self, pad_name: str, metal: pya.Region, edge: pya.Region,
                   other_edge: pya.Region) -> None:
        pads = self.region(pad_name)
        prefix = "I" if pad_name.startswith("I") else ""
        for polygon in pads.each_merged():
            bbox = polygon.bbox()
            if not polygon.is_box():
                self.add(prefix + "BP.1", "Wirebond pad marker must be rectangular", polygon)
            if min(bbox.width(), bbox.height()) < self.u(100):
                self.add(prefix + "BP.WL.1", "Wirebond pad width and length >= 100 um", polygon)
        self.add_region(prefix + "BP.2", "Pad purpose must cover base-metal mask purpose",
                        pads - metal)
        if not edge.is_empty():
            edge_band = edge - edge.sized(-self.u(200))
            self.add_region(prefix + "BP.CE.E.1", "Wirebond pad must interact with 200 um ChipEdge band",
                            pads.not_interacting(edge_band))
        if not other_edge.is_empty():
            self.add_pairs(prefix + "BP.ICE.S.1", "Wirebond pad space to bonded chip edge >= 50 um",
                           pads.separation_check(other_edge, self.u(50)))

    def run(self) -> int:
        # SQP 3.0 section 2: anything outside the published layer/purpose map
        # is reserved and may be stripped during aggregation.
        known_layers = {layer for layer, _datatype in LAYERS.values()} | {98, 99}
        base_layers = {5, 10, 105, 110}
        manufactured = {1, 3, 5, 10, 20, 30, 31, 40, 45, 55,
                        101, 103, 105, 110, 120, 130, 131, 140, 145, 155}
        for index in list(self.layout.layer_indices()):
            info = self.layout.get_info(index)
            if info.layer in {98, 99}:
                allowed = info.datatype == 228
            else:
                allowed = (info.layer in known_layers and
                           (info.datatype in {0, 228} or
                            (info.layer in base_layers and info.datatype in {2, 11}) or
                            (info.layer in manufactured and info.datatype in {237, 251})))
            if not allowed:
                unexpected = pya.Region(self.cell.begin_shapes_rec(index))
                self.add_region("LAYER.RESERVED", f"Reserved layer/purpose {info.layer}/{info.datatype}", unexpected)

        q_edge, i_edge = self.region("CHIP_EDGE"), self.region("ICHIP_EDGE")
        if q_edge.is_empty() and i_edge.is_empty():
            marker = self.cell.bbox()
            if marker.empty():
                marker = pya.Box(-1, -1, 1, 1)
            self.add_box("CE.1", "A required ChipEdge layer (703 or 713) is missing", marker)
        self.check_edge("QCE", q_edge)
        self.check_edge("ICE", i_edge)
        if not q_edge.is_empty() and not i_edge.is_empty():
            interacting = q_edge.interacting(i_edge)
            self.add_region("QCE.ICE.E.1", "Interacting QChipEdge needs 500 um IChipEdge enclosure",
                            interacting - i_edge.sized(-self.u(500)))

        q_m1 = self.base_region("M1"); q_m2 = self.base_region("M2")
        i_m1 = self.base_region("IM1"); i_m2 = self.base_region("IM2")
        for name, edge in (("M1", q_edge), ("M2", q_edge), ("IM1", i_edge), ("IM2", i_edge)):
            self.check_base_metal(name, edge)
        for a, b, prefix in ((q_m1, q_m2, "Q"), (i_m1, i_m2, "I")):
            if not a.is_empty() and not b.is_empty():
                self.add_pairs(prefix + "MET.MET.S.1", "Different base metals minimum space 5 um",
                               a.separation_check(b, self.u(5)))
                overlap = (a & b).merged()
                self.check_min_area(prefix + "MET.MET.A.2", "Base-metal overlap area >= 50 um^2", overlap, 50)

        q_se = self.union("SE1", "SEB1"); i_se = self.union("ISE1", "ISEB1")
        self.check_posts("POST", "POSTCLEAN", q_m1 | q_m2, i_m1 | i_m2,
                         q_se, i_se, self.region("SEB1"), self.region("BR"),
                         self.region("IBR"), self.region("UBM"), self.region("IUBM"),
                         q_edge, i_edge)
        self.check_posts("IPOST", "IPOSTCLEAN", i_m1 | i_m2, q_m1 | q_m2,
                         i_se, q_se, self.region("ISEB1"), self.region("IBR"),
                         self.region("BR"), self.region("IUBM"), self.region("UBM"),
                         i_edge, q_edge)

        self.check_se("SE1", "SEB1", "SE1_DEVICE", q_m1 | q_m2, q_edge)
        self.check_se("ISE1", "ISEB1", "ISE1_DEVICE", i_m1 | i_m2, i_edge)
        self.check_bridges("BR", "RR", "IBR", q_m1 | q_m2 | q_se, q_edge)
        self.check_bridges("IBR", "IRR", "BR", i_m1 | i_m2 | i_se, i_edge)
        self.check_ubm_bumps("UBM", "BUMP", q_m1 | q_m2, q_edge)
        self.check_ubm_bumps("IUBM", "IBUMP", i_m1 | i_m2, i_edge)
        self.check_pads("M1PAD", q_m1, q_edge, i_edge)
        self.check_pads("M2PAD", q_m2, q_edge, i_edge)
        self.check_pads("IM1PAD", i_m1, i_edge, q_edge)
        self.check_pads("IM2PAD", i_m2, i_edge, q_edge)

        unsupported = [
            "POST.2 force-distribution symmetry (manual by specification)",
            "full Dolan finger decomposition: SE.DJ.1-.4 and DFM lengths",
            "array-JJ topology: SE.DA.1 and window pairing",
            "net-aware electrical overlap rules SE.SE.O.1/O.2",
            "bridge crossover/tether bin extraction BR.3/BR.4/BR.WL.1",
            "BUMP.2 redundancy (manual by specification)",
            "process alignment-cross integrity and foundry auto-waivers",
        ]
        self.notes.extend(unsupported)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.rdb.save(str(self.report_path))
        summary_path = self.report_path.with_suffix(".txt")
        total = sum(self.counts.values())
        with summary_path.open("w", encoding="utf-8") as stream:
            stream.write(f"SQUILL SQP 3.0 KLayout preflight v{VERSION}\n")
            stream.write(f"Cell: {self.cell.name}\nDBU: {self.dbu} um\n")
            stream.write(f"Mode: {'production + DFM advisory' if self.dfm else 'production'}\n")
            stream.write(f"Violations: {total}\n\n")
            for rule, count in sorted(self.counts.items()):
                if count:
                    stream.write(f"{rule}: {count}\n")
            stream.write("\nNot implemented; use official Calibre signoff:\n")
            for note in self.notes:
                stream.write(f"- {note}\n")
        print(f"SQUILL preflight: {total} marker(s)")
        print(f"Marker database: {self.report_path}")
        print(f"Text summary: {summary_path}")
        return total


def _load_layout():
    input_path = globals().get("input")
    requested_top = globals().get("top")
    app = pya.Application.instance()
    main_window = app.main_window() if app is not None else None
    view = main_window.current_view() if main_window is not None else None
    active_cv = view.active_cellview() if view is not None else None
    if input_path:
        # If this file is already open, use its viewer-owned Layout so markers
        # can navigate and zoom to the original polygons.
        if active_cv is not None and active_cv.is_valid() and active_cv.filename():
            try:
                same_file = Path(active_cv.filename()).resolve() == Path(str(input_path)).resolve()
            except OSError:
                same_file = False
            if same_file:
                return (active_cv.layout(), active_cv.cell,
                        Path(active_cv.filename()), view, active_cv.index())
        # In interactive KLayout, make an explicitly supplied input viewer-owned
        # as well.  Batch KLayout has no MainWindow and follows the path below.
        if main_window is not None:
            loaded_cv = main_window.load_layout(
                str(input_path), pya.LoadLayoutOptions(), "", 1
            )
            if requested_top:
                selected = loaded_cv.layout().cell(str(requested_top))
                if selected is None:
                    raise RuntimeError(f"Top cell {requested_top!r} was not found")
                loaded_cv.cell = selected
            loaded_view = main_window.current_view()
            loaded_view.add_missing_layers()
            loaded_view.max_hier()
            loaded_view.zoom_fit()
            return (loaded_cv.layout(), loaded_cv.cell, Path(str(input_path)),
                    loaded_view, loaded_cv.index())
        layout = pya.Layout()
        layout.read(str(input_path))
        if requested_top:
            cell = layout.cell(str(requested_top))
            if cell is None:
                raise RuntimeError(f"Top cell {requested_top!r} was not found")
        else:
            tops = list(layout.top_cells())
            if len(tops) != 1:
                names = ", ".join(cell.name for cell in tops)
                raise RuntimeError(f"Expected one top cell, found {len(tops)}: {names}; pass -rd top=NAME")
            cell = tops[0]
        return layout, cell, Path(str(input_path)), None, None

    if active_cv is None or not active_cv.is_valid():
        raise RuntimeError("No input=... supplied and no active KLayout cellview")
    return (active_cv.layout(), active_cv.cell,
            Path(active_cv.filename() or f"{active_cv.cell.name}.gds"),
            view, active_cv.index())


def main() -> int:
    layout, cell, source, view, cv_index = _load_layout()
    output = globals().get("report")
    report_path = Path(str(output)) if output else source.with_name(source.stem + "_sqp_preflight.lyrdb")
    checker = SQPPreflight(layout, cell, report_path,
                           dfm=_runtime_bool("dfm", False),
                           local_checks=_runtime_bool("local", True))
    if abs(layout.dbu - EXPECTED_DBU_UM) > 1e-12:
        marker = cell.bbox() if not cell.bbox().empty() else pya.Box(-1, -1, 1, 1)
        checker.add_box("PRECISION.1", f"GDS DBU must be {EXPECTED_DBU_UM} um (2000 dbu/um), found {layout.dbu}", marker)
    total = checker.run()
    if view is not None and _runtime_bool("show", True):
        # Make sure a layout loaded during macro start-up is visibly selected.
        cell_view = view.cellview(cv_index)
        cell_view.cell = cell
        view.add_missing_layers()
        view.max_hier()
        view.zoom_fit()
        rdb_index = view.add_rdb(checker.rdb)
        view.show_rdb(rdb_index, cv_index)
        print("Marker Browser opened: select a rule and double-click a marker to zoom/highlight it")
    return total


if __name__ == "__main__":
    main()
