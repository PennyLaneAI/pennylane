# Template gallery artwork

These illustrations are used by `doc/introduction/templates.rst`. They are
conceptual thumbnails, not circuit diagrams or complete decompositions.
Keep detailed figures used by the API documentation in their existing locations.

Each template has a self-contained SVG named after its public API class. The
gallery directive copies SVGs directly, so no image generation tool or extra
documentation dependency is required.

When adding or editing an illustration:

- Use a `400 × 280` view box, the same pale background, and a generous clear margin.
- Keep outlines at 4 units with rounded caps and joins. Check the result at the
  gallery's displayed size, approximately `196 × 128` pixels.
- Use `#151515` for outlines, `#0074E0` for blue accents, `#D5F0FD` for light blue,
  `#FFB5F1` for pink, and `#FFE08A` for occasional warm accents. The black, pink,
  and light blue match PennyLane's circuit-drawing style; the blue matches its
  documentation theme.
- Keep the central motif specific to the template. Prefer a few clear shapes
  over small labels, equations, or a full circuit.
- Include a descriptive SVG `title` and `desc`. These describe the standalone
  artwork; the gallery's visible caption remains the link to the API.
- Keep the SVG editable and self-contained: no scripts, external references,
  embedded raster images, or font dependencies.

To preview the gallery, build the documentation with `make docs` from the
repository root and open `doc/_build/html/introduction/templates.html`.
