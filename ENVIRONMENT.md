# Conda environment

This repository includes exported Conda environment files for the `marl` environment.

## Recreate the environment

Use the full exported file for the closest match:

```bash
conda env create -f environment.yml
conda activate marl
```

If the environment already exists:

```bash
conda env update -n marl -f environment.yml --prune
conda activate marl
```

`environment-history.yml` contains only the manually requested Conda specs and is useful as a lightweight reference.
`environment-linux-lock.yml` is a second full export kept for Linux-to-Linux reproduction.

