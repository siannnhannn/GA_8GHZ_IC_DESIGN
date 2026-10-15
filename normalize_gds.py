"""Flatten and union a GDS library into one fabrication-ready cell."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gdstk
from shapely import unary_union
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.validation import make_valid


def normalize_gds(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cell_name: str = "SPIN_CHAIN",
    datatype: int = 0,
    precision: float = 5e-7,
    chip_edge_layer: int = 703,
    reserve_id_corner: bool = True,
) -> None:
    """Flatten *input_path*, union polygons per layer, and write *output_path*.

    ``precision`` is expressed in the GDS user unit.  This design is exported
    in millimetres, so 5e-7 mm is the foundry's 0.5 nm database grid.  Using a
    0.5 um boolean grid here would close the intentional submicron SQUID gaps.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = gdstk.read_gds(input_path)
    top_cells = source.top_level()
    if len(top_cells) != 1:
        names = ", ".join(cell.name for cell in top_cells)
        raise ValueError(f"Expected one top-level cell, found: {names}")

    polygons_by_layer: dict[int, list[Polygon]] = defaultdict(list)
    for polygon in top_cells[0].get_polygons(
        apply_repetitions=True, include_paths=True, depth=None
    ):
        # Qiskit Metal's positive-mask renderer puts the resolved fabricated
        # base metal on layer 5/datatype 0.  Datatypes 10 and 11 are duplicate
        # polygon/path geometry views; unioning them back into datatype 0 fills
        # CPW isolation gaps and shorts signal traces to the ground plane.
        if polygon.layer == 5 and polygon.datatype != 0:
            continue
        polygons_by_layer[polygon.layer].append(
            make_valid(Polygon(polygon.points))
        )

    # SQP 2.7 requires 2000 database units per micrometre (0.5 nm DBU).
    result = gdstk.Library(unit=source.unit, precision=5e-10)
    output_cell = result.new_cell(cell_name)
    source_bounds = top_cells[0].bounding_box()
    for layer, polygons in sorted(polygons_by_layer.items()):
        # Bridge layers must not be grid-snapped (SQP 2.6); snapping rotated
        # RR/BR rectangles deforms their exact dimensions.  Other layers use
        # the 0.5 nm database grid so submicron SQUID/JJ features survive.
        merged = (
            unary_union(polygons)
            if layer in (30, 31)
            else unary_union(polygons, grid_size=precision)
        )
        if layer == 5 and reserve_id_corner:
            # SQP CE.2 reserves the upper-left 300 x 100 um for foundry ID.
            min_corner, max_corner = source_bounds
            id_corner = box(
                min_corner[0],
                max_corner[1] - 0.100,
                min_corner[0] + 0.300,
                max_corner[1],
            )
            if isinstance(merged, list):
                merged = [shape.difference(id_corner) for shape in merged]
            else:
                merged = merged.difference(id_corner)
        merged_polygons = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
        for polygon in merged_polygons:
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            exterior = gdstk.Polygon(
                polygon.exterior.coords,
                layer=layer,
                datatype=datatype,
            )
            holes = [
                gdstk.Polygon(interior.coords, layer=layer, datatype=datatype)
                for interior in polygon.interiors
            ]
            if holes:
                output_cell.add(
                    *gdstk.boolean(
                        [exterior],
                        holes,
                        "not",
                        precision=precision,
                        layer=layer,
                        datatype=datatype,
                    )
                )
            else:
                output_cell.add(exterior)

    # SQP CE.1 requires an orthogonal ChipEdge rectangle centered at (0, 0).
    min_corner, max_corner = source_bounds
    output_cell.add(
        gdstk.rectangle(
            min_corner,
            max_corner,
            layer=chip_edge_layer,
            datatype=datatype,
        )
    )

    # Do not fracture connected polygons during serialization.  Modern mask
    # tools support boundaries beyond the historical 199-vertex convention.
    result.write_gds(output_path, max_points=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-name", default="SPIN_CHAIN")
    parser.add_argument("--datatype", type=int, default=0)
    args = parser.parse_args()
    normalize_gds(
        args.input,
        args.output,
        cell_name=args.cell_name,
        datatype=args.datatype,
    )


if __name__ == "__main__":
    main()
