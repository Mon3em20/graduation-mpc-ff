# MPC programs

The repository is the authoritative home for the two implemented `.mpc`
sources. MP-SPDZ is an external installation and is not a Python dependency.
The current programs are isolated one-example fixed-point BP checks, not
end-to-end MNIST training:

- `programs/bp/bp_tiny.mpc` checks a 2-2-1 secret network, one forward pass,
  backpropagation, and one SGD update.
- `programs/bp/relu_edge.mpc` checks the inactive and active ReLU branches.

Set `MP_SPDZ_ROOT` to an existing MP-SPDZ checkout and run, for example:

```bash
MP_SPDZ_ROOT="$HOME/MP-SPDZ" \
MP_SPDZ_PYTHON="$HOME/venvs/mp-spdz-ml/bin/python" \
./scripts/run_mpc.sh bp_tiny.mpc
```

The runner copies only the explicitly requested authoritative source into
`Programs/Source`, compiles with `-M -E ring`, and records output under
`results/mpc/bp/`. It does not modify the MP-SPDZ framework source or keys.
The three-party semi-honest Rep3 Ring setup and `sfix` precision are
provisional project settings. Compilation is not by itself a security or
training verification.
