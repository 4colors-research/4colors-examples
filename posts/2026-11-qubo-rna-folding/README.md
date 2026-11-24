# RNA folding QUBOs, refereed: the model the papers posed, solved and checked exactly

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-rna-folding&utm_content=readme-top)

## What you will learn

- How an RNA sequence becomes candidate stems, and stems become QUBO variables (one per stem).
- How the paper's energy (stacking reward, average stem length, a softer reward for crossing stems) becomes one
  integer matrix.
- How to replace the paper's "infinite penalty" for overlapping stems with exact rows, one per shared base, so no
  penalty weight is needed and every constrained answer is checked.
- How to referee the answers: an exact branch and bound proves the optimum on small cases, and the script says
  plainly when it gives up.

## The model

A stem m = (i, j, k) is k stacked pairs (i, j), (i+1, j−1), …, (i+k−1, j−k+1), each A-U, G-C or G-U. As in the
paper, every stem of at least 3 pairs is a candidate, sub-stems included, and a hairpin loop keeps at least 2
unpaired bases. Variable q_m is 1 when stem m is chosen. The paper's Eq 13 is

```
H(q) = sum_m [ c_L (k_m^2 - 2 k_m mu + mu^2) - c_B k_m^2 ] q_m  -  sum_{m<n} c_B k_m k_n delta_mn q_m q_n
```

with mu the length of the longest candidate stem, delta_mn = c_P when stems m and n cross (a pseudoknot) and 1
otherwise, and c_B = 1, c_L = 10, c_P = 0.5, the paper's values. The c_L term pulls the chosen stems toward the
longest possible length; the quadratic term rewards stems that can coexist, crossing ones at half weight.

The paper forbids two stems that share a base with an infinite penalty. A penalty has to be finite in practice, and
choosing it is the usual QUBO headache. Here the rule is a set of exact rows instead:

"Two stems overlap" means exactly "they share a base", so one row per base, Σ q_m ≤ 1 over the stems that use it,
admits exactly the same structures as one "not both" row q_m + q_n ≤ 1 per overlapping pair, with far fewer rows.
`--rows pairs` sends the literal pairwise form instead; the tests check that both admit exactly the
non-overlapping sets.

Two details keep the numbers exact. With c_P = 0.5 the energy has halves, so the model holds 2H and every
coefficient is an integer; scripts report H. And the quadratic term is written only for stems that do not overlap:
an overlapping pair can never both be chosen, so its term never counts in a feasible answer.

## Data

- The paper's worked example, GGAAGCAAACAUCCCUGU (18 nt; Fox et al. 2022).
- Seeded random sequences (`rna.random_sequence(length, seed)`) at 30, 45 and 60 nt, the size range of the hardware
  studies. Random sequences fold into nothing biological; they are test inputs for the QUBO with a known size.
- Your own: `--fasta my.fasta` or `--sequence ACGU...`. Use sequences whose terms you have checked. Nothing is
  downloaded.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-rna-folding&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-rna-folding&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: the paper's example

```bash
python run.py --out runs/example --example --time-limit 1 --seeds 1
```

The script lists the candidate stems, computes the exact optimum, solves once through the API, and re-scores the
answer from its 0/1 values with Eq 13. A line reads like:

## Step 2: the hardware-study sizes, refereed

```bash
python run.py --out runs/rand --lengths 30 45 60 --instances 1 2 3 4 5 --time-limit 1 --seeds 1 2 3
```

Fifteen random sequences, three solver seeds each. For each sequence the exact reference is a branch and bound over
sets of compatible stems: every quadratic term is negative or zero, which gives a simple valid lower bound, and the
search stops at `--exact-max-nodes` (2 million by default). If it stops, the row says `gave_up`, the instance gets no
gap, and the post does not call anything optimal for it.

## Step 3: plot

```bash
python plot.py runs/example runs/rand --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the stem enumerator on hand-worked hairpins (loop length and G-U pairs included), the overlap and
crossing tests, the model's energy against Eq 13 on every feasible assignment of a small case, that both row forms
admit exactly the non-overlapping stem sets, the exact branch and bound against brute force, the dot-bracket
writer with a pseudoknot, the FASTA reader, and the run and plot scripts on fixed hand-made answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
