# Manim scenes

Animated and still figures built with Manim Community (v0.18+).

* The build environment has no LaTeX, so scenes use `Text`, not `MathTex`/`Tex`.
* Render a still frame (used in the book):
  `manim -qm -s --format=png manim/scenes/<file>.py <SceneName>`
* Render a short GIF: `manim -ql --format=gif manim/scenes/<file>.py <SceneName>`
* Rendered outputs live in `docs/assets/figures/`; the `media/` folder is ignored.
