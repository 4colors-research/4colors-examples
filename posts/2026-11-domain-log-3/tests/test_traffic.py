"""Unit tests of the route-assignment model on hand-made inputs. Nothing here calls the service.

    python -m pytest -q tests
"""
import itertools
from pathlib import Path

import traffic

# Two cars, two routes each, sharing segment (1, 2) on their first routes.
ROUTES = [
    [((0, 1), (1, 2)), ((0, 3), (2, 3))],
    [((1, 2), (2, 5)), ((1, 4), (4, 5))],
]

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="10" lat="52.0000" lon="5.0000"/>
  <node id="11" lat="52.0010" lon="5.0000"/>
  <node id="12" lat="52.0010" lon="5.0010"/>
  <node id="13" lat="52.0500" lon="5.0500"/>
  <node id="14" lat="52.0510" lon="5.0500"/>
  <node id="15" lat="52.0005" lon="5.0005"/>
  <way id="1"><nd ref="10"/><nd ref="11"/><nd ref="12"/><tag k="highway" v="residential"/></way>
  <way id="2"><nd ref="13"/><nd ref="14"/><tag k="highway" v="primary"/></way>
  <way id="3"><nd ref="10"/><nd ref="15"/><tag k="highway" v="footway"/></way>
  <way id="4"><nd ref="12"/><nd ref="10"/><tag k="highway" v="service"/><tag k="access" v="private"/></way>
</osm>
"""


def test_energy_equals_congestion_on_every_valid_assignment():
    m = traffic.build_model(ROUTES)
    assert m.n == 4 and m.m == 2
    for choice in itertools.product(range(2), range(2)):
        x = traffic.encode(ROUTES, choice)
        assert m.feasible(x)
        assert m.energy(x) == traffic.congestion(ROUTES, choice)
    assert traffic.congestion(ROUTES, (0, 0)) == 1 + 4 + 1          # segment (1, 2) carries two cars
    assert traffic.congestion(ROUTES, (1, 0)) == 4


def test_rows_reject_zero_or_two_routes():
    m = traffic.build_model(ROUTES)
    assert not m.feasible([0, 0, 1, 0]) and not m.feasible([1, 1, 1, 0])
    assert traffic.decode(ROUTES, [1, 1, 1, 0]) is None
    assert traffic.decode(ROUTES, [0, 1, 1, 0]) == [1, 0]


def test_same_car_pairs_are_left_out():
    _, quad = traffic.qubo_terms(ROUTES)
    assert quad == {(0, 2): 2}                                       # only the cross-car shared segment


def test_grid_city_routes_and_alternatives():
    G, xy = traffic.grid_city(6, 6, arterial_every=3)
    assert len(G) == 36 and sum(len(v) for v in G.values()) // 2 == 2 * 6 * 5
    routes = traffic.candidate_routes(G, 0, 35, k=3)
    assert 1 <= len(routes) <= 3 and len(set(routes)) == len(routes)
    lengths = [traffic.route_length(G, r) for r in routes]
    assert lengths[0] == min(lengths)                                # route 0 is the shortest
    for r in routes:                                                 # each route is a path from 0 to 35
        nodes = [n for e in r for n in e]
        assert nodes.count(0) == 1 and nodes.count(35) == 1


def test_best_response_never_worsens_and_reaches_equilibrium():
    G, _ = traffic.grid_city(8, 8)
    cars = traffic.random_cars(G, 40, seed=3)
    routes = [traffic.candidate_routes(G, s, t, 3) for s, t in cars]
    start = traffic.all_shortest(routes)
    br, rounds = traffic.best_response(routes, start)
    assert traffic.congestion(routes, br) <= traffic.congestion(routes, start)
    again, r2 = traffic.best_response(routes, br)
    assert again == br and r2 == 1                                   # nobody moves at an equilibrium


def test_count_terms_matches_the_model():
    G, _ = traffic.grid_city(6, 6)
    routes = [traffic.candidate_routes(G, s, t, 3) for s, t in traffic.random_cars(G, 12, seed=1)]
    size = traffic.count_terms(routes)
    m = traffic.build_model(routes)
    assert size["n_vars"] == m.n and size["n_rows"] == m.m == 12
    assert size["n_quad"] == m.nnz


def test_osm_parser_keeps_drivable_largest_component(tmp_path):
    p = Path(tmp_path) / "tiny.osm"
    p.write_text(OSM)
    G, xy = traffic.osm_graph(str(p))
    assert len(G) == 3                                               # 10-11-12; 13-14 is a smaller component
    assert sorted(len(v) for v in G.values()) == [1, 1, 2]           # footway and private service dropped
    d = sorted(L for u in G for L in G[u].values())
    assert 67 < d[0] < 70 and 110 < d[-1] < 112      # 0.001 deg of longitude at 52 N is about 68.5 m; of latitude 111 m


def test_safe_lambda_beats_dropping_a_car():
    import rival

    lam = rival.safe_lambda(ROUTES)
    assert lam == 1 + max(sum(2 * c + 1 for c in cs) for cs in ([1, 2], [1, 1], [2, 1], [1, 1]))
    bqm = rival.penalty_bqm(ROUTES, lam)
    energies = {bits: bqm.energy(dict(enumerate(bits))) for bits in itertools.product([0, 1], repeat=4)}
    best = min(energies, key=energies.get)
    choice = traffic.decode(ROUTES, best)
    assert choice is not None and traffic.congestion(ROUTES, choice) == 4   # one route per car, optimal
    assert energies[best] == 4
