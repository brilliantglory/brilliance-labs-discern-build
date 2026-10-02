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

def cyborg_skull():
    """Original cyborg-skull icon: half bone, half machine."""
    cranium = scale(Point(0, 2.5).buffer(1, 128), 8.5, 8.0, origin=(0, 2.5))
    jaw = box(-4.2, -8.3, 4.2, -1).buffer(1.3, 32)
    head = unary_union([cranium, jaw]).buffer(-0.8).buffer(1.6).buffer(-0.8)
    outline = head.difference(head.buffer(-1.1))
    # viewer-left eye: organic socket
    eye_l = scale(Point(-3.4, 0.4).buffer(1, 64), 2.5, 2.1, origin=(-3.4, 0.4))
    # viewer-right eye: angular machine socket with raised "lens" island
    eye_r = Polygon([(1.2, 2.3), (5.6, 2.3), (6.0, 0.4), (5.2, -1.6), (1.6, -1.6), (1.0, 0.4)])
    lens = Point(3.5, 0.35).buffer(0.95, 48)
    eye_r = eye_r.difference(lens)
    nose = Polygon([(-1.0, -2.3), (1.0, -2.3), (0, -4.3)])
    mouth_y0, mouth_y1 = -7.6, -5.0
    teeth = [LineString([(-3.6, -6.3), (3.6, -6.3)]).buffer(0.4, cap_style=2)]
    for x in (-2.4, -0.8, 0.8, 2.4):
        teeth.append(LineString([(x, mouth_y0), (x, mouth_y1)]).buffer(0.4, cap_style=2))
    # machine plating seam + rivets on the right half of the cranium
    seam = LineString([(0.9, 10.0), (0.9, 6.2), (4.0, 4.4), (7.3, 4.6)]).buffer(0.4, join_style=2)
    rivets = [Point(p).buffer(0.55, 24) for p in [(2.6, 7.6), (4.6, 6.6), (6.0, 7.9)]]
    icon = unary_union([outline, eye_l, eye_r, nose, seam, *teeth, *rivets])
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
        shape = cyborg_skull() if label == "SKULL" else letter(label)
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
