# Unverified citations queue

Chapter authors have a finite WebSearch budget. When it runs out before every
source has been verified, the author does NOT guess a URL (STYLE.md section 4
forbids it). Instead they cite by title and venue, and record the citation here
so a later pass with fresh budget can convert it into a verified link.

One file per part, named `<part-dir>.md`, each line in the form:

    - [ ] docs/part04-vision/04-detection.md | Focal Loss for Dense Object Detection | Lin et al., ICCV 2017 | believed arXiv:1708.02002

A citations editor verifies each line, adds the link to the chapter, and ticks
the box. A line that cannot be verified is marked `- [x] UNVERIFIABLE` with a
note, and the chapter keeps its title-and-venue citation.
