# Tutorial material for trame v2

trame - a web framework that weaves together open source components into customized visual analytics easily.

Please use the links below to find more informations:
- [Tutorial](https://kitware.github.io/trame/guide/tutorial/)
- [trame website](https://kitware.github.io/trame/)

## Run the tomography webapp

Entry point: `my_tests/tomo_viewer.py`

From the repository root:

```bash
pixi install
pixi run python my_tests/tomo_viewer.py --port 8080
```

Then open:

```text
http://localhost:8080
```

Optional: auto-load a TIFF file on startup:

```bash
pixi run python my_tests/tomo_viewer.py --port 8080 --data /absolute/path/to/your_file.tif
```

Notes:

- Input data should be `.tif` or `.tiff`.
- If you see missing package errors for `numpy` or `matplotlib`, add them with:

```bash
pixi add numpy matplotlib
```
