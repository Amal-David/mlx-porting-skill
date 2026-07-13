const body = document.body;
const pauseButton = document.querySelector("[data-pause]");
const restartButton = document.querySelector("[data-restart]");

function setPaused(paused) {
  body.classList.toggle("is-paused", paused);
  pauseButton?.setAttribute("aria-pressed", String(paused));
  if (pauseButton) pauseButton.textContent = paused ? "Resume" : "Pause";
}

pauseButton?.addEventListener("click", () => {
  setPaused(!body.classList.contains("is-paused"));
});

restartButton?.addEventListener("click", () => {
  setPaused(false);
  body.classList.add("is-restarting");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => body.classList.remove("is-restarting"));
  });
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) setPaused(true);
});
