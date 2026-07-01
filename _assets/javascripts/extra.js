// minimal extras
document.addEventListener("DOMContentLoaded", () => {
  // 给所有外链加 ↗
  document.querySelectorAll(".md-content a[href^='http']").forEach((a) => {
    if (a.querySelector("img")) return;
    if (a.textContent.includes("↗")) return;
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener");
  });
});
