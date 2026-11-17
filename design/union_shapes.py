"""UnionShapes: merge the geometry of several components into one shape.

Takes the qgeometry of the listed components, turns their paths into polygons
(rounded with each path's fillet, as the renderers draw them), and unions
everything that shares a layer, chip and polarity: all the metal becomes one
polygon and all the etch another. The source components' own rows are then
removed from the qgeometry tables, so only the merged shape is rendered and
exported. The source components stay in the design (pins, options,
get_points(), airbridge placement all still work).

Create it AFTER the components it merges (design.rebuild() rebuilds in
creation order, and the sources must be rebuilt before this one). If you
re-create a source component, re-create this one after it.

Qiskit Metal's GDS renderer can't export a MultiPolygon row, so if a union
has pieces that don't touch (e.g. the Purcell filter, which only couples
capacitively to the readout resonator) each piece is its own row
(<name>_0, <name>_1, ...), with the largest first.
"""

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from qiskit_metal import Dict
from qiskit_metal.qlibrary.core import QComponent

from airbridge_route import _fillet_centerline


class UnionShapes(QComponent):
    default_options = Dict(
        components="",          # comma-separated component names to merge
        chip="main",
    )

    component_metadata = Dict(
        short_name="UnionShapes",
        _qgeometry_table_poly="True",
    )

    # touching shapes can miss each other by floating-point noise; grow each
    # shape by this much before the union and shrink the result back (0.1 nm)
    _EPS = 1e-7  # mm

    def source_names(self):
        return [n.strip() for n in str(self.options.components).split(",") if n.strip()]

    def make(self):
        qgeom = self.design.qgeometry
        ids = []
        for name in self.source_names():
            if name not in self.design.components:
                raise ValueError(f"UnionShapes {self.name}: no component named '{name}'")
            ids.append(self.design.components[name].id)

        groups = {}   # (layer, chip, subtract) -> list of polygons
        for table in ("poly", "path"):
            df = qgeom.tables[table]
            df = df[df.component.isin(ids)]
            for _, row in df.iterrows():
                if row.get("helper", False):
                    continue
                geom = row.geometry
                if table == "path":
                    pts = np.array(geom.coords, dtype=float)
                    fillet = row.get("fillet", 0)
                    if fillet and not np.isnan(fillet) and len(pts) > 2:
                        pts = _fillet_centerline(pts, float(fillet))
                    geom = LineString(pts).buffer(
                        float(row["width"]) / 2, cap_style=2, join_style=2)
                # row["subtract"], not row.subtract: that is pandas' Series.subtract method
                key = (str(row["layer"]), str(row["chip"]), bool(row["subtract"]))
                groups.setdefault(key, []).append(geom.buffer(self._EPS, join_style=2))

        # the sources' own rows go; only the merged shape is rendered
        for cid in ids:
            qgeom.delete_component_id(cid)

        for (layer, chip, subtract), geoms in groups.items():
            merged = unary_union(geoms).buffer(-self._EPS, join_style=2)
            parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
            parts = sorted((g for g in parts if isinstance(g, Polygon) and not g.is_empty),
                           key=lambda g: -g.area)
            kind = "etch" if subtract else "metal"
            names = ([f"{kind}_L{layer}"] if len(parts) == 1
                     else [f"{kind}_L{layer}_{i}" for i in range(len(parts))])
            self.add_qgeometry(
                "poly",
                dict(zip(names, parts)),
                layer=int(layer) if layer.isdigit() else layer,
                subtract=subtract,
                chip=chip,
            )
