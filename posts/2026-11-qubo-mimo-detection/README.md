# MIMO detection as a QUBO: from QuAMax to a classical solver, measured in milliseconds

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-mimo-detection&utm_content=readme-top)

## What you will learn

- How ML detection for QPSK and 16-QAM becomes an Ising model: real-valued form, amplitudes as spins
  (x = 2 s1 + s2 for 16-QAM), a dense coupling matrix with no penalty terms.
- How to send it with `fourcolors.qubo.solve` and read back the bits, including the Gray-code transform.
- How to score each answer the same way, against MMSE and against the exact ML answer from a sphere decoder.
- How to read the time to the ML answer from the solve's trajectory, and how to pack many channel uses into one
  job.

## The model

Write the complex system in real form: y_r = [Re y; Im y] and H_r = [[Re H, -Im H], [Im H, Re H]]. Each real
symbol component is an amplitude: +-1 for QPSK, and +-1 or +-3 for 16-QAM. Two spins per 16-QAM amplitude
(x = 2 s1 + s2) and one per QPSK amplitude give x_r = T s, and then

```
||y_r - H_r T s||^2 = s' G s - 2 b' s + ||y_r||^2,     G = T' H_r' H_r T,   b = T' H_r' y_r
```

With s_i^2 = 1 the diagonal of G is a constant, so the Ising model is

```
E(s) = sum_{i<j} 2 G_ij s_i s_j  -  2 sum_i b_i s_i          (plus a constant: ||y_r||^2 + trace G)
```

That is the whole model: N_t log2(M) spins (64 for 32 users with QPSK, 128 for 32 users with 16-QAM), dense

The solver returns spins in natural binary order (+3 is (+1, +1)); constellations are Gray-labeled, so neighboring
points differ in one bit. `spins_to_gray_bits` is the fixed transform between the two, which QuAMax calls the
QAM-to-QUBO transform.

## Data

All channels are generated: i.i.d. Rayleigh fading (unit-variance complex Gaussian entries), uniformly random data
bits, and complex Gaussian noise at a given SNR (average received symbol energy per antenna over noise variance).
Every channel use is a pure function of its seed, so you get the same instances we did. Nothing is downloaded.
i.i.d. Rayleigh is the standard textbook channel; correlated or measured channels change the numbers (see the
limits).

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-mimo-detection&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-mimo-detection&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: one channel use, three detectors

```bash
python run.py --out runs/try --nt 8 --mod qpsk --snr 8 --frames 2 --time-limit 1
```

For each channel use the script runs MMSE and zero-forcing (local, a few microseconds of linear algebra), a
Schnorr-Euchner sphere decoder (local, exact ML when it finishes, with a count of the tree nodes it visited), and
one 4colors job. Each answer goes through the same scorer, `mimo.score`: bit errors on the Gray bits, symbol errors,

`is_ml` says whether the answer has the sphere decoder's residual. `t_to_ml_s` is the search time at which the
service's best answer first reached it, read from the job's trajectory: that is the quality-per-millisecond number.

## Step 2: bit error rate on square systems

```bash
python run.py --out runs/ber-q32 --nt 32 --mod qpsk --snr 4 8 12 16 --frames 200 --pack 50 --time-limit 1
python run.py --out runs/ber-h16 --nt 16 --mod 16qam --snr 12 16 20 24 --frames 200 --pack 50 --time-limit 1
```

A bit error rate needs many channel uses. `--pack 50` sends 50 channel uses as one block-diagonal model (one job,
50 independent blocks) and decodes each block on its own. The blocks do not interact, so this is the same 50
problems at a lower per-job overhead; the time limit is then shared by the 50 blocks.

## Step 3: time to the ML answer as the system grows

```bash
for nt in 8 16 24 32 40 48; do
  python run.py --out runs/ttm-q$nt --nt $nt --mod qpsk --snr 4 --frames 10 --time-limit 1 --seeds 1 2 3
done
```

One job per channel use here, so each `t_to_ml_s` belongs to one problem. The sphere decoder's node count on the

## Step 4: plot

```bash
python plot.py runs/ber-* runs/ttm-* --out figures --budget 1
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check, on every spin assignment of small cases, that the Ising energy plus the constant equals
||y - Hx||^2 for QPSK and 16-QAM; that the sphere decoder matches brute-force ML; that MMSE and zero-forcing
recover the symbols on a noiseless orthogonal channel; that bits, spins and amplitudes round-trip and the 16-QAM
labels are Gray; that a packed model is block-diagonal; the script's rows on fixed hand-made answers; and the
charts' selection rule on hand-made rows.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
