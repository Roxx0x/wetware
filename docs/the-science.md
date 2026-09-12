# The science

Nothing here is invented. `wetware` is three established results bolted together.

## 1. The brain is real and complete

The larval *Drosophila* connectome (Winding et al., *Science* 2023) is the first
synapse-resolution wiring diagram of an entire animal brain: every neuron, every
connection, mapped by electron microscopy. 2952 neurons in the all-to-all matrix,
~110,000 weighted directed connections, weight = number of synapses. This is the
`W` you load. It is not a model of a brain; it is a brain, digitised.

- Winding et al., *The connectome of an insect brain*, Science 2023 — https://www.science.org/doi/10.1126/science.add9330
- Data: https://github.com/brain-networks/larval-drosophila-connectome

## 2. A fixed recurrent network can compute — echo-state networks

Reservoir computing (echo-state networks; liquid state machines) showed that you
don't need to train a recurrent network to use it. A *fixed* random recurrent
network, driven by input, produces a rich high-dimensional trace of its own
activity. Train a single linear layer to read that trace and you can solve real
temporal tasks. The one condition is the **echo-state property**: the network must
forget its initial state, which you get by scaling the recurrent weights so the
spectral radius sits just below ~1.

`wetware` scales the connectome to that regime and leaves everything else as
measured.

## 3. The connectome makes a good reservoir — the "Biological Processing Unit"

Recent work replaces the *random* reservoir with a *connectome*: use the animal's
real wiring as the fixed recurrent core, train only input/output. It works, and
the biological structure brings its own properties (e.g. resilience to
overfitting).

- Biological Processing Units: Leveraging an Insect Connectome — https://arxiv.org/pdf/2507.10951
- The Connectome of a Fly as a Computational Reservoir (ESA ACT) — https://www.esa.int/gsp/ACT/projects/fly_connectome/
- The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12109256/

## How the three fit together in this repo

```
connectome W  ─►  scale to echo-state regime  ─►  drive with input u[t]
                                                        │
                                             x[t] = (1-a)x[t-1] + a·tanh(W x[t-1] + Win u[t])
                                                        │
                                             collect states X  ─►  ridge readout  Wout = (XᵀX+λI)⁻¹ XᵀY
```

- `reservoir.py` — steps 1–3 (scale, drive, collect).
- `readout.py` — the closed-form ridge solve. The only trained parameters.
- `data.py` — fetch and cache the real `W`.

## What we add, and disclose

- **Signs.** The published matrix is unsigned synapse counts. We assign a fraction
  of neurons an inhibitory sign so the reservoir has E/I dynamics. Modelling
  choice, not measurement.
- **Spectral scaling.** Standard ESN practice; changes the global gain, not who
  connects to whom.

Everything else — the wiring, the weights — is the measured brain.

## What it is not

It is not a simulation of fly behaviour, and the readout is not biological. The
claim is exactly: *a real, complete animal connectome, used as a reservoir,
carries enough structure that a linear readout computes non-trivial tasks from
it.* That claim is true and reproducible with `wetware run digits`.
