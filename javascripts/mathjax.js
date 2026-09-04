window.MathJax = {
tex: {
  // arithmatex converts $...$ and $$...$$ in source Markdown to these delimiters.
  inlineMath: [["\\(", "\\)"]],
  displayMath: [["\\[", "\\]"]],
  processEscapes: true,
  processEnvironments: true
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
