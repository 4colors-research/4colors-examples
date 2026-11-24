"""Unit tests of the MATPOWER reader, the partition models, the checks and the script plumbing (no service calls)."""
import itertools

import examplekit
import grid as gr
import plot
import rival_sa
import run

# Two triangles (buses 1-2-3 and 4-5-6) joined by branch 3-4; a parallel 1-2 branch; an out-of-service 1-6 branch.
CASE = """function mpc = toy
mpc.version = '2';
mpc.baseMVA = 100.0;
%% bus data
mpc.bus = [
	1	3	0.0	0.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
	2	1	20.0	5.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
	3	1	0.0	0.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
	4	1	10.0	0.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
	5	1	0.0	0.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
	6	2	0.0	0.0	0.0	0.0	1	1.0	0.0	230.0	1	1.1	0.9;
];
%% branch data
mpc.branch = [
	1	2	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	1	2	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	2	3	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	1	3	0.01	0.2	0.0	100	100	100	0	0	1	-30	30;
	3	4	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;  % the tie
	4	5	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	5	6	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	4	6	0.01	0.1	0.0	100	100	100	0	0	1	-30	30;
	1	6	0.01	0.1	0.0	100	100	100	0	0	0	-30	30;  % out of service
];
"""


def graph():
    return gr.to_graph(gr.parse_matpower(CASE), "unit")


def test_reader_merges_parallel_and_drops_out_of_service():
    g = graph()
    assert g["n"] == 6 and len(g["edges"]) == 7
    assert dict(zip(g["edges"], g["w"]))[(0, 1)] == 2         # two parallel 1-2 branches
    assert (0, 5) not in g["edges"]
    s = gr.to_graph(gr.parse_matpower(CASE), "susceptance")
    assert dict(zip(s["edges"], s["w"]))[(0, 2)] == 1           # x = 0.2 is half as stiff as the median


def test_bisection_energy_is_cut_and_row_is_balance():
    g = graph()
    c = gr.work(g, "unit")
    m = gr.bisection_model(g, c, 0.0)
    for x in itertools.product([0, 1], repeat=6):
        assert abs(m.energy(list(x)) - gr.cut_weight(g, x)) < 1e-9
        assert bool(m.feasible(list(x))) == (sum(x) == 3) == gr.balanced(x, c, 2, 0.0)


def test_kway_energy_is_cut_on_one_hot_assignments():
    g = graph()
    c = gr.work(g, "degree")
    m = gr.kway_model(g, c, 3, 0.5)
    for part in itertools.product(range(3), repeat=6):
        x = [0] * 18
        for i, p in enumerate(part):
            x[i * 3 + p] = 1
        assert gr.decode(x, 6, 3) == list(part)
        assert abs(m.energy(x) - gr.cut_weight(g, part)) < 1e-9
        assert bool(m.feasible(x)) == gr.balanced(part, c, 3, 0.5)
    assert gr.decode([1, 1, 0] + [0] * 15, 6, 3) is None


def test_bounds_and_work():
    assert gr.bounds(10, 2, 0.0) == (5, 5)
    assert gr.bounds(10, 2, 0.1) == (5, 5)          # 4.5..5.5 on integers
    assert gr.bounds(100, 4, 0.03) == (25, 25)
    assert gr.bounds(101, 2, 0.03) == (49, 52)
    assert gr.work(graph(), "degree") == [1 + 2, 1 + 2, 1 + 3, 1 + 3, 1 + 2, 1 + 2]


def test_milp_reference_finds_the_tie():
    g = graph()
    ref = gr.milp_reference(g, gr.work(g, "unit"), 2, 0.0, time_limit_s=10)
    assert ref["status"] == "optimal" and ref["cut"] == 1
    assert sorted(ref["part"][:3]) == [ref["part"][0]] * 3       # each triangle stays together


def test_rival_penalty_bqm_and_short_anneal():
    g = graph()
    c = gr.work(g, "unit")
    bqm = rival_sa.penalty_bqm(g, c, 0.5)
    for x in itertools.product([0, 1], repeat=6):
        want = gr.cut_weight(g, x) + 0.5 * (sum(x) - 3) ** 2
        assert abs(bqm.energy(dict(enumerate(x))) - want) < 1e-9
    res = rival_sa.anneal(g, c, 0.0, 0.5, 0.05, seed=1, sweeps=50, batch=2)
    assert res["reads"] > 0 and (res["cut"] is None or res["cut"] >= 1)


def test_run_end_to_end_and_plots(canned, tmp_path):
    case = tmp_path / "toy_case6.m"
    case.write_text(CASE)
    canned.solution = lambda call: [1, 1, 1, 0, 0, 0] if call["n"] == 6 else \
        [v for p in (0, 0, 1, 1, 2, 2) for v in [int(p == q) for q in range(3)]]
    run.main(["--out", str(tmp_path / "g"), "--case", str(case), "--parts", "2", "3", "--imbalance", "0.0",
              "--budgets", "1", "--seeds", "1", "2"])
    rows = examplekit.Results(tmp_path / "g" / "results.jsonl").rows()
    assert [r["buses"] for r in rows if r["kind"] == "instance"] == [6]
    refs = {r["parts"]: r for r in rows if r["kind"] == "reference"}
    assert refs[2]["status"] == "optimal" and refs[2]["cut"] == 1
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 4
    bis = [r for r in solves if r["parts"] == 2]
    assert all(r["cut"] == 1 and r["balanced"] and r["model_feasible"] and r["service_feasible"] for r in bis)
    assert all(abs(r["energy_python"] - r["solver_objective"]) < 1e-9 for r in solves)
    three = [r for r in solves if r["parts"] == 3][0]
    assert three["part"] == [0, 0, 1, 1, 2, 2] and three["cut"] == gr.cut_weight(graph(), [0, 0, 1, 1, 2, 2])
    rival = tmp_path / "s" / "results.jsonl"
    examplekit.Results(rival).add(kind="rival", case="toy_case6", time_limit_s=1.0, cut=3, tuning=False)
    plot.main([str(tmp_path / "g"), str(tmp_path / "s"), "--out", str(tmp_path / "fig"), "--draw", "toy_case6:2",
               "--case-file", str(case)])
    for f in ("grid_partition.png", "grid_gap.png", "grid_vs_sa.png"):
        assert (tmp_path / "fig" / f).exists()
    cells, won = plot.vs_rival(examplekit.load_rows([tmp_path / "g", tmp_path / "s"]))
    assert len(cells) == 1 and len(won) == 1


def test_rival_refuses_dense_penalty(tmp_path):
    case = tmp_path / "toy_case6.m"
    case.write_text(CASE)
    rival_sa.main(["--out", str(tmp_path / "r"), "--case", str(case), "--max-buses", "5"])
    row = examplekit.Results(tmp_path / "r" / "results.jsonl").rows()[0]
    assert "skipped" in row
