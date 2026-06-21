# berkeley_hackathon

This build keeps the existing monument test cases and adds an improved **Custom monument PLY upload** workflow.

The core reconstruction pipeline remains unchanged:

```text
python/completion_pipeline.py
```

## Included subjects

```text
Broken Roman Arena
Broken Palmyra Arch
Demolished Leaning Tower of Pisa
Custom monument (upload PLY)
```

## Built-in test cases

Each built-in monument includes four deterministic test cases, including the absurd non-symmetric half-chop demolition case.

```text
Case 1 - local demolition case
Case 2 - second demolition case
Case 3 - diagonal/severe damage case
Case 4 - Absurd half-chop demolition
```

## New custom upload control

When you select:

```text
Test subject -> Custom monument (upload PLY)
```

the panel now shows:

```text
Custom reconstruction mode:
- Conservative
- Balanced
- Aggressive
```

Use them like this:

```text
Conservative = safest, fewer red particles
Balanced     = recommended default
Aggressive   = fills larger missing regions, may add more red particles
```

For leaning or slanted custom objects, Balanced and Aggressive use an adaptive height-wise centerline before symmetry reflection. This makes uploaded towers or slanted monuments reconstruct better than the older rigid vertical-plane method.

## Supported PLY files

```text
ASCII PLY with x y z vertices
Binary little-endian PLY with x y z vertices
Binary big-endian PLY with x y z vertices
```

## Run

```bash
cd "C:\Hrishikesh\point cloud project\multi-monument-completion-custom-upload-modes"
py -m pip install -r python/requirements.txt
py python/run_subject.py --all --regenerate
py -m http.server 8000
```

Open:

```text
http://localhost:8000/index.html
```

Then choose:

```text
Test subject -> Custom monument (upload PLY)
Custom reconstruction mode -> Balanced or Aggressive
Upload your .ply file
View -> Reconstructed particles or Completed cloud
```

Changing the custom mode reprocesses the uploaded PLY in the browser. It does not overwrite Roman Arena, Palmyra Arch, or Leaning Tower outputs.

## Notes

Custom PLY reconstruction is generic. It does not know the monument type, so results depend on how symmetric and clean the uploaded point cloud is. Balanced is the best default for most uploaded objects. Use Aggressive when the red reconstruction is too sparse.
