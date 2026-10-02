"""Generate the Hopeful.AI promo die (engraved symbols on a rounded cube)."""
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString, box
from shapely.affinity import scale, translate
from shapely.ops import unary_union
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import manifold3d as m3d
import trimesh

SIZE = 30.0      # die edge length, mm
CORNER_R = 3.0   # corner/edge rounding radius, mm
DEPTH = 1.0      # engraving depth, mm
CAP = 17.0       # letter cap height, mm
FONT = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

def text_shape(s):
    path = TextPath((0, 0), s, size=100, prop=FONT)
    geom = Polygon()
    for ring in path.to_polygons(closed_only=True):
        if len(ring) >= 3:
            geom = geom.symmetric_difference(Polygon(ring).buffer(0))  # even-odd fill
    return geom

H_HEIGHT = text_shape("H").bounds[3] - text_shape("H").bounds[1]

def center(g):
    x0, y0, x1, y1 = g.bounds
    return translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)

def letter(s, cap=CAP, max_w=21.0):
    g = text_shape(s)
    k = cap / H_HEIGHT
    w = (g.bounds[2] - g.bounds[0]) * k
    if w > max_w:
        k *= max_w / w
    return center(scale(g, k, k, origin=(0, 0)))

def mirror_x(pts):
    """Close a left-right symmetric outline from its right half (top to bottom)."""
    return pts + [(-x, y) for x, y in reversed(pts) if x != 0]

def robot_skull():
    """Original menacing robot-skull icon: angular head, angry brow, glowing lenses."""
    head = Polygon(mirror_x([
        (0, 10.5), (3.6, 10.3), (6.6, 9.0), (8.4, 6.4), (8.9, 3.4), (8.5, 0.8),
        (8.0, -1.4), (6.7, -3.0), (6.6, -5.6), (6.0, -8.6), (4.4, -10.1), (0, -10.5),
    ]))
    outline = head.difference(head.buffer(-1.0, join_style=2))
    # angry, angled sockets (dark) with raised glowing lenses
    sock_r = Polygon([(1.0, 1.9), (6.4, 3.5), (7.0, 1.5), (6.2, -0.6), (2.0, -0.9), (0.8, 0.3)])
    socks = unary_union([sock_r, scale(sock_r, -1, 1, origin=(0, 0))])
    lenses = unary_union([Point(x, 1.15).buffer(1.15, 48) for x in (3.9, -3.9)])
    eyes = socks.difference(lenses)
    brow = LineString([(-8.6, 5.3), (0, 3.0), (8.6, 5.3)]).buffer(0.45, join_style=2)
    nose = Polygon([(0, -1.6), (1.3, -3.7), (0.55, -4.4), (0, -3.9), (-0.55, -4.4), (-1.3, -3.7)])
    cheeks = unary_union([LineString([(s * 6.8, -1.6), (s * 2.6, -2.9)]).buffer(0.4, cap_style=2)
                          for s in (1, -1)])
    # clenched mechanical teeth: dark mouth with two rows of block teeth left raised
    mouth = Polygon([(-3.6, -4.9), (3.6, -4.9), (3.2, -9.0), (-3.2, -9.0)])
    inner = mouth.buffer(-0.5, join_style=2)
    teeth, n, gap = [], 5, 0.55
    x0, x1 = -3.1, 3.1
    w = (x1 - x0 - (n - 1) * gap) / n
    for y0, y1 in ((-6.65, -5.4), (-8.5, -7.25)):
        for i in range(n):
            xa = x0 + i * (w + gap)
            teeth.append(box(xa, y0, xa + w, y1))
    mouth_cut = mouth.difference(unary_union(teeth).intersection(inner))
    # hydraulic jaw pistons
    pistons = []
    for s in (1, -1):
        pistons.append(LineString([(s * 5.2, -3.4), (s * 4.7, -7.8)]).buffer(0.4, cap_style=2))
        pistons += [Point(s * 5.2, -3.4).buffer(0.7, 32), Point(s * 4.7, -7.8).buffer(0.7, 32)]
    bolts = [Point(s * 4.6, 7.6).buffer(0.6, 24) for s in (1, -1)]
    icon = unary_union([outline, eyes, brow, nose, cheeks, mouth_cut, *pistons, *bolts])
    return center(icon.intersection(head))

def to_cross_section(g):
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    rings = []
    for p in polys:
        p = shapely.geometry.polygon.orient(p, 1.0)
        rings.append(np.asarray(p.exterior.coords)[:-1])
        rings += [np.asarray(i.coords)[:-1] for i in p.interiors]
    return m3d.CrossSection(rings, m3d.FillRule.EvenOdd)

def rounded_cube():
    h = SIZE / 2 - CORNER_R
    s = m3d.Manifold.sphere(CORNER_R, 48)
    corners = [s.translate((x, y, z)) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
    return m3d.Manifold.batch_hull(corners)

# face: (normal, up) — H P F L wrap the sides to read "HoPeFuL", .AI on top, skull on bottom
FACES = [
    ("H",    (0, -1, 0), (0, 0, 1)),
    ("P",    (1, 0, 0),  (0, 0, 1)),
    ("F",    (0, 1, 0),  (0, 0, 1)),
    ("L",    (-1, 0, 0), (0, 0, 1)),
    (".AI",  (0, 0, 1),  (0, 1, 0)),
    ("SKULL", (0, 0, -1), (0, -1, 0)),
]

def build():
    die = rounded_cube()
    cutters = []
    for label, n, u in FACES:
        n, u = np.array(n, float), np.array(u, float)
        r = np.cross(u, n)
        shape = robot_skull() if label == "SKULL" else letter(label)
        cut = m3d.Manifold.extrude(to_cross_section(shape), DEPTH + 1.0).translate((0, 0, -DEPTH))
        c = n * SIZE / 2
        M = np.column_stack([r, u, n, c])  # local (x,y,z) -> world
        cutters.append(cut.transform(M))
    die = die - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
    mesh = die.to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "hopeful_ai_die.stl"
    m = build()
    m.export(out)
    print(out, "watertight:", m.is_watertight, "tris:", len(m.faces), "bounds:", m.bounds.tolist())
