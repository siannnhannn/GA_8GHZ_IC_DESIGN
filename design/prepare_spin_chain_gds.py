"""Prepare the spin-chain mask and enforce its airbridge landing rules."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gdstk
import numpy as np
from shapely import affinity, unary_union
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
from shapely.validation import make_valid

from airbridge_route import (
    _arclength_frame,
    _collapse_uturn_corners,
    _fillet_centerline,
    _sample_at,
)
from normalize_gds import normalize_gds


RR_ENCLOSURE = 0.0015  # 1.5 um in the design's millimetre user unit
BR_SPACING = 0.005  # 5 um
CORNER_SEARCH_STEP = 0.001  # 1 um along the bend
CORNER_SEARCH_LIMIT = 0.090  # can traverse a full 50 um-radius 180-degree turn


def _polygon(gds_polygon: gdstk.Polygon):
    return make_valid(Polygon(gds_polygon.points))


def _element_polygons(element):
    """Yield polygons from a Qiskit renderer qgeometry GDS element."""
    if element is None:
        return
    # ``chip_info`` has used several container shapes across Qiskit Metal
    # releases: raw GDS objects, dictionaries containing ``geometry`` and
    # nested lists.  Keep this adapter deliberately permissive so signal
    # geometry is not silently dropped during layer-5 reconstruction.
    if isinstance(element, dict):
        for value in element.values():
            yield from _element_polygons(value)
        return
    if isinstance(element, gdstk.Cell):
        yield from element.get_polygons(
            apply_repetitions=True, include_paths=True, depth=None
        )
        return
    if isinstance(element, gdstk.Polygon):
        yield element
    elif isinstance(element, (gdstk.FlexPath, gdstk.RobustPath)):
        yield from element.to_polygons()
    elif hasattr(element, "to_polygons"):
        yield from element.to_polygons()
    elif hasattr(element, "geometry"):
        yield from _element_polygons(element.geometry)
    elif isinstance(element, (list, tuple)):
        for child in element:
            yield from _element_polygons(child)


def _bridge_geometry(x: float, y: float, orientation: float):
    """Return the Bin-2 RR pair and 12-vertex BR dogbone at a placement."""

    crossover_length = 0.025
    bridge_width = 0.0075
    pad_length = 0.0135
    pad_width = 0.0135
    rr_length = 0.010
    rr_width = 0.0105
    pad_offset = crossover_length / 2 + pad_length / 2

    rr = [
        box(-pad_offset - rr_length / 2, -rr_width / 2,
            -pad_offset + rr_length / 2, rr_width / 2),
        box(+pad_offset - rr_length / 2, -rr_width / 2,
            +pad_offset + rr_length / 2, rr_width / 2),
    ]
    pads = [
        box(-pad_offset - pad_length / 2, -pad_width / 2,
            -pad_offset + pad_length / 2, pad_width / 2),
        box(+pad_offset - pad_length / 2, -pad_width / 2,
            +pad_offset + pad_length / 2, pad_width / 2),
    ]
    span = box(
        -(crossover_length + pad_length) / 2,
        -bridge_width / 2,
        +(crossover_length + pad_length) / 2,
        +bridge_width / 2,
    )

    def place(shape):
        return affinity.translate(
            affinity.rotate(shape, orientation, origin=(0, 0)), x, y
        )

    return [place(shape) for shape in rr], unary_union(
        [place(shape) for shape in [*pads, span]]
    )


def _to_gdstk(shape, layer: int):
    polygons = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    result = []
    for polygon in polygons:
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        exterior = gdstk.Polygon(
            polygon.exterior.coords, layer=layer, datatype=0
        )
        holes = [
            gdstk.Polygon(interior.coords, layer=layer, datatype=0)
            for interior in polygon.interiors
        ]
        if holes:
            result.extend(
                gdstk.boolean(
                    [exterior], holes, "not", precision=5e-7,
                    layer=layer, datatype=0,
                )
            )
        else:
            result.append(exterior)
    return result


def _elements_union(elements):
    """Resolve a renderer chip-info collection to one Shapely geometry."""
    return unary_union(
        [
            _polygon(polygon)
            for element in elements
            for polygon in _element_polygons(element)
        ],
        grid_size=5e-7,
    )


def prepare_spin_chain_gds(
    input_path: str | Path,
    output_path: str | Path,
    waveguide_points,
    *,
    turn_radius: float = 0.050,
    layer5_signal_elements=None,
    layer5_subtract_elements=None,
    reserved_points=None,
    reserved_radius: float = 0.090,
):
    """Clean invalid bridges, restore every waveguide-turn bridge, and flatten."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    library = gdstk.read_gds(input_path)
    top = library.top_level()
    if len(top) != 1:
        raise ValueError("The source GDS must contain one top-level design")

    by_layer = defaultdict(list)
    for polygon in top[0].get_polygons(True, True, None):
        by_layer[polygon.layer].append(polygon)

    # Rebuild base metal with the renderer's actual qgeometry polarity.  The
    # resolved datatype-0 view can contain solid ground across some routed CPWs
    # (notably the emitter charge/flux feeds).  Adding only subtract=False
    # center traces cannot repair that view because their isolation gaps have
    # already been filled.  Apply the route clearances first, then restore the
    # positive center conductors: ground - subtract=True + subtract=False.
    # Raw GDS datatypes 10/11 are merely Polygon versus LineString source views
    # and therefore cannot be used to infer this polarity.
    resolved_layer5 = unary_union(
        [
            _polygon(polygon)
            for polygon in by_layer[5]
            if polygon.datatype == 0
        ],
        grid_size=5e-7,
    )
    if layer5_subtract_elements is not None:
        resolved_layer5 = resolved_layer5.difference(
            _elements_union(layer5_subtract_elements)
        )
    if layer5_signal_elements is not None:
        resolved_layer5 = resolved_layer5.union(
            _elements_union(layer5_signal_elements)
        )
    by_layer[5] = _to_gdstk(make_valid(resolved_layer5), 5)

    # Datatype 0 is Qiskit Metal's resolved positive-mask result.  The layer-5
    # datatype 10/11 polygons are source-geometry views and must not be treated
    # as ground when validating RR post landings.
    metal = unary_union(
        [
            _polygon(polygon)
            for polygon in by_layer[5]
            if polygon.datatype == 0
        ],
        # Coordinates are millimetres: use the 0.5 nm database grid so the
        # bridge landing check cannot gain false enclosure from coarse snap.
        grid_size=5e-7,
    )
    safe_ground = metal.buffer(-RR_ENCLOSURE, join_style="mitre")
    rr = [_polygon(polygon) for polygon in by_layer[30]]
    br = [_polygon(polygon) for polygon in by_layer[31]]

    br_tree = STRtree(br)
    rr_owner = {}
    owned_rr = defaultdict(list)
    for rr_index, post in enumerate(rr):
        candidates = br_tree.query(post.centroid, predicate="within")
        if len(candidates) == 0:
            raise ValueError(f"RR post {rr_index} has no enclosing BR dogbone")
        owner = min(
            map(int, candidates),
            key=lambda index: br[index].centroid.distance(post.centroid),
        )
        rr_owner[rr_index] = owner
        owned_rr[owner].append(rr_index)

    # Snap every exported dogbone (and its two RR feet) to the permitted
    # Manhattan/diagonal orientation lattice.  Some renderer-generated
    # bridges arrive with small residual rotations from fillet sampling even
    # after route placement is snapped.
    for bridge_index, bridge in enumerate(br):
        rect = bridge.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        edges = [
            (coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
            for i in range(4)
        ]
        dx, dy = max(edges, key=lambda edge: edge[0] ** 2 + edge[1] ** 2)
        angle = np.degrees(np.arctan2(dy, dx))
        correction = round(angle / 45.0) * 45.0 - angle
        if abs(correction) < 1e-6:
            continue
        origin = bridge.centroid
        br[bridge_index] = affinity.rotate(bridge, correction, origin=origin, use_radians=False)
        for rr_index in owned_rr[bridge_index]:
            rr[rr_index] = affinity.rotate(rr[rr_index], correction, origin=origin, use_radians=False)

    failing_br = {
        rr_owner[index]
        for index, post in enumerate(rr)
        if not safe_ground.covers(post)
    }
    # A usable crossover must retain exactly one grounded landing post at
    # each end.  Route edits can leave an obsolete span after both associated
    # posts have been rejected; do not carry such an orphan into the mask.
    failing_br.update(
        index for index in range(len(br)) if len(owned_rr[index]) != 2
    )
    # SQP BR.O.1.DFM: remove bridges that do not cross a base-metal opening.
    failing_br.update(
        index for index, bridge in enumerate(br) if metal.covers(bridge)
    )

    centerline, corner_lengths = _fillet_centerline(
        waveguide_points, turn_radius, return_corners=True
    )
    frame = _arclength_frame(centerline)
    corner_lengths, uturn_lengths = _collapse_uturn_corners(
        frame, corner_lengths, turn_radius, return_uturns=True
    )
    # Turns inside the route's airbridge keepout zones (e.g. the waveguide's
    # qubit pocket and coupling_fillet corners) intentionally carry no bridge,
    # so they are neither restored nor required to have an apex bridge.
    if reserved_points is not None and len(reserved_points):
        reserved = np.asarray(reserved_points, dtype=float)[:, :2]

        def _is_reserved(length):
            x, y, _ = _sample_at(frame, length)
            return bool(
                np.any(np.hypot(reserved[:, 0] - x, reserved[:, 1] - y)
                       < reserved_radius)
            )

        corner_lengths = [s for s in corner_lengths if not _is_reserved(s)]
        uturn_lengths = [s for s in uturn_lengths if not _is_reserved(s)]
    corner_owners = []
    for length in corner_lengths:
        x, y, _ = _sample_at(frame, length)
        corner_owners.append(
            min(
                range(len(br)),
                key=lambda index: br[index].centroid.distance(Point(x, y)),
            )
        )

    # Enforce BR.S.1 before restoring mandatory turn bridges.  Prefer keeping
    # a waveguide-turn bridge over a uniform or manually placed bridge.
    protected_all = set(corner_owners)
    spacing_tree = STRtree(br)
    spacing_removals = set()
    for index, bridge in enumerate(br):
        if index in failing_br or index in spacing_removals:
            continue
        for other in map(int, spacing_tree.query(bridge.buffer(BR_SPACING))):
            if other <= index or other in failing_br or other in spacing_removals:
                continue
            if bridge.distance(br[other]) >= BR_SPACING - 1e-9:
                continue
            if index in protected_all and other in protected_all:
                raise ValueError(
                    "Two mandatory waveguide-turn bridges violate BR spacing"
                )
            spacing_removals.add(index if other in protected_all else other)
    failing_br.update(spacing_removals)

    protected_owners = set(corner_owners) - failing_br
    keep_br = [
        (index, shape, index in protected_owners)
        for index, shape in enumerate(br)
        if index not in failing_br
    ]
    keep_rr = [
        (rr_owner[index], shape)
        for index, shape in enumerate(rr)
        if rr_owner[index] not in failing_br
    ]
    relocated = []

    for corner_index, (length, owner) in enumerate(
        zip(corner_lengths, corner_owners)
    ):
        if owner not in failing_br:
            continue

        found = None
        # A U-turn crossover belongs exactly at the semicircle apex.  Ordinary
        # 90-degree bends may still move along the arc to find valid grounded
        # landings, but moving a U-turn bridge produces the visibly off-center
        # placement this cleanup is meant to prevent.
        offsets = [0.0]
        if length not in uturn_lengths:
            steps = int(CORNER_SEARCH_LIMIT / CORNER_SEARCH_STEP)
            for step in range(1, steps + 1):
                offsets.extend(
                    (-step * CORNER_SEARCH_STEP, step * CORNER_SEARCH_STEP)
                )

        for offset in offsets:
            x, y, orientation = _sample_at(frame, length + offset)
            candidate_rr, candidate_br = _bridge_geometry(x, y, orientation)
            if not all(safe_ground.covers(post) for post in candidate_rr):
                continue
            conflicts = [
                index
                for index, (_, other, _) in enumerate(keep_br)
                if candidate_br.distance(other) < BR_SPACING
            ]
            if any(keep_br[index][2] for index in conflicts):
                continue
            found = (candidate_rr, candidate_br, offset, conflicts)
            break

        if found is None:
            raise ValueError(
                f"No DRC-valid bridge placement found on waveguide turn {corner_index}"
            )
        candidate_rr, candidate_br, offset, conflicts = found
        conflicting_owners = {keep_br[index][0] for index in conflicts}
        keep_br = [
            record for index, record in enumerate(keep_br) if index not in conflicts
        ]
        keep_rr = [
            record for record in keep_rr if record[0] not in conflicting_owners
        ]
        relocation_owner = -(corner_index + 1)
        keep_rr.extend((relocation_owner, post) for post in candidate_rr)
        keep_br.append((relocation_owner, candidate_br, True))
        relocated.append((corner_index, offset))

    # Every retained post must have the foundry-required base-metal enclosure.
    if not all(safe_ground.covers(post) for _, post in keep_rr):
        raise AssertionError("Airbridge cleanup retained an invalid RR post")

    # U-turn bridges must remain at the exact arc apex after invalid bridge and
    # spacing cleanup.  One nanometre accommodates the GDS database grid while
    # still catching a bridge shifted to either side of the turn.
    retained_bridges = [shape for _, shape, _ in keep_br]
    for length in uturn_lengths:
        x, y, _ = _sample_at(frame, length)
        distance = min(
            bridge.centroid.distance(Point(x, y))
            for bridge in retained_bridges
        )
        if distance > 1e-6:
            raise AssertionError(
                f"Waveguide U-turn bridge is {distance * 1e3:.3f} um "
                "from its apex"
            )

    working = output_path.with_suffix(".working.gds")
    cleaned = gdstk.Library(unit=library.unit, precision=library.precision)
    cell = cleaned.new_cell("SPIN_CHAIN_CLEANED")
    for layer, polygons in by_layer.items():
        if layer in (30, 31):
            continue
        cell.add(*polygons)
    for _, post in keep_rr:
        cell.add(*_to_gdstk(post, 30))
    for _, bridge, _ in keep_br:
        cell.add(*_to_gdstk(bridge, 31))
    cleaned.write_gds(working, max_points=0)
    normalize_gds(working, output_path)
    working.unlink()

    return {
        "input_bridges": len(br),
        "removed_invalid_bridges": len(failing_br),
        "relocated_waveguide_turn_bridges": relocated,
        "output_bridges": len(keep_br),
        "rr_posts": len(keep_rr),
        "waveguide_turns": len(corner_lengths),
        "waveguide_uturns": len(uturn_lengths),
    }
