window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      R: "\\mathbb{R}",
      E: "\\mathbb{E}",
      KL: "D_{\\mathrm{KL}}",
      softmax: "\\operatorname{softmax}",
      argmax: "\\operatorname{arg\\,max}",
      argmin: "\\operatorname{arg\\,min}",
      tr: "\\operatorname{tr}",
      diag: "\\operatorname{diag}",
      norm: ["\\left\\lVert #1 \\right\\rVert", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
