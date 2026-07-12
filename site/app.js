(() => {
  "use strict";

  document.documentElement.classList.add("js");

  const data = window.MLX_PORTING_SITE_DATA || null;
  const repository = "https://github.com/Amal-David/mlx-porting-skill";

  const get = (path) => {
    if (!data) return undefined;
    return path.split(".").reduce((value, key) => (value == null ? undefined : value[key]), data);
  };

  document.querySelectorAll("[data-value]").forEach((node) => {
    const value = get(node.dataset.value || "");
    if (value !== undefined && value !== null) node.textContent = String(value);
  });

  document.querySelectorAll("[data-year]").forEach((node) => {
    node.textContent = String(new Date().getFullYear());
  });

  const portLoop = document.querySelector("[data-port-loop]");
  const portLoopToggle = portLoop?.querySelector("[data-port-loop-toggle]");
  portLoopToggle?.addEventListener("click", () => {
    const paused = portLoop.classList.toggle("is-paused");
    portLoopToggle.setAttribute("aria-pressed", String(paused));
    portLoopToggle.textContent = paused ? "Resume motion" : "Pause motion";
  });

  const renderFamilies = (container, compact = false) => {
    const families = get("architectures.families");
    if (!container || !Array.isArray(families) || families.length === 0) return;
    container.replaceChildren();
    families.forEach((family, index) => {
      const article = document.createElement(compact ? "div" : "article");
      article.className = compact ? "route-node" : "family-card";
      if (compact) {
        const title = document.createElement("a");
        title.href = `${repository}/blob/main/mlx-model-porting/${family.runbook}`;
        title.textContent = family.label;
        title.rel = "noopener";
        article.append(title);
      } else {
        const indexLabel = document.createElement("span");
        indexLabel.className = "card-index";
        indexLabel.textContent = String(index + 1).padStart(2, "0");
        article.append(indexLabel);
        const heading = document.createElement("h3");
        heading.textContent = family.label;
        article.append(heading);
      }
      const id = document.createElement("span");
      id.className = compact ? "" : "route-tag";
      id.textContent = family.id;
      article.append(id);
      if (!compact) {
        const link = document.createElement("a");
        link.className = "route-tag";
        link.href = `${repository}/blob/main/mlx-model-porting/${family.runbook}`;
        link.textContent = "Open runbook ↗";
        link.rel = "noopener";
        article.append(link);
      }
      container.append(article);
    });
  };

  renderFamilies(document.querySelector("#family-grid"));
  renderFamilies(document.querySelector("#docs-route-map"), true);

  const menuButton = document.querySelector("[data-nav-toggle]");
  if (menuButton) {
    menuButton.addEventListener("click", () => {
      const open = document.body.classList.toggle("nav-open");
      menuButton.setAttribute("aria-expanded", String(open));
    });
    document.querySelectorAll(".site-header .nav-links a").forEach((link) => {
      link.addEventListener("click", () => {
        document.body.classList.remove("nav-open");
        menuButton.setAttribute("aria-expanded", "false");
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("nav-open")) {
        document.body.classList.remove("nav-open");
        menuButton.setAttribute("aria-expanded", "false");
        menuButton.focus();
      }
    });
  }

  const docsMenuButton = document.querySelector("[data-docs-nav-toggle]");
  const docsSidebar = document.querySelector("#docs-sidebar");
  const docsMain = document.querySelector("#docs-main");
  const docsCloseButton = document.querySelector("[data-docs-nav-close]");
  const docsBackdrop = document.querySelector("[data-docs-backdrop]");
  const docsMobile = window.matchMedia("(max-width: 860px)");

  if (docsMenuButton && docsSidebar) {
    const isOpen = () => document.body.classList.contains("docs-nav-open");
    const focusableInSidebar = () => [...docsSidebar.querySelectorAll(
      "a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex='-1'])",
    )].filter((node) => node instanceof HTMLElement && !node.hidden && getComputedStyle(node).visibility !== "hidden");

    const setDocsMenu = (open, { moveFocus = false, returnFocus = false } = {}) => {
      if (!docsMobile.matches) {
        document.body.classList.remove("docs-nav-open");
        docsMenuButton.setAttribute("aria-expanded", "false");
        docsSidebar.removeAttribute("aria-hidden");
        docsSidebar.removeAttribute("inert");
        docsMain?.removeAttribute("inert");
        return;
      }
      document.body.classList.toggle("docs-nav-open", open);
      docsMenuButton.setAttribute("aria-expanded", String(open));
      docsSidebar.toggleAttribute("inert", !open);
      if (open) {
        docsSidebar.removeAttribute("aria-hidden");
        docsMain?.setAttribute("inert", "");
        if (moveFocus) (docsCloseButton || focusableInSidebar()[0])?.focus();
      } else {
        docsSidebar.setAttribute("aria-hidden", "true");
        docsMain?.removeAttribute("inert");
        if (returnFocus) docsMenuButton.focus();
      }
    };

    docsMenuButton.addEventListener("click", () => {
      setDocsMenu(!isOpen(), { moveFocus: !isOpen(), returnFocus: isOpen() });
    });
    docsCloseButton?.addEventListener("click", () => setDocsMenu(false, { returnFocus: true }));
    docsBackdrop?.addEventListener("click", () => setDocsMenu(false, { returnFocus: true }));
    document.querySelectorAll(".docs-nav a").forEach((link) => {
      link.addEventListener("click", () => {
        const targetId = (link.getAttribute("href") || "").slice(1);
        const target = targetId ? document.getElementById(targetId) : null;
        const heading = target?.querySelector("h2");
        const moveToTarget = docsMobile.matches && heading instanceof HTMLElement;
        setDocsMenu(false);
        if (moveToTarget) {
          heading.setAttribute("tabindex", "-1");
          window.requestAnimationFrame(() => {
            heading.focus({ preventScroll: true });
            heading.addEventListener("blur", () => heading.removeAttribute("tabindex"), { once: true });
          });
        }
      });
    });
    document.addEventListener("keydown", (event) => {
      if (!docsMobile.matches || !isOpen()) return;
      if (event.key === "Escape") {
        const searchField = document.querySelector("#docs-search");
        if (document.activeElement === searchField && searchField.value) return;
        event.preventDefault();
        setDocsMenu(false, { returnFocus: true });
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = focusableInSidebar();
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      } else if (!docsSidebar.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      }
    });
    const syncDocsMenu = () => setDocsMenu(false);
    if (typeof docsMobile.addEventListener === "function") docsMobile.addEventListener("change", syncDocsMenu);
    else docsMobile.addListener(syncDocsMenu);
    syncDocsMenu();
  }

  let copyIndex = 0;
  document.querySelectorAll("pre[data-copy]").forEach((pre) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy command");
    const status = document.createElement("span");
    status.className = "sr-only copy-status";
    status.id = `copy-status-${copyIndex += 1}`;
    status.setAttribute("aria-live", "polite");
    status.setAttribute("aria-atomic", "true");
    button.setAttribute("aria-describedby", status.id);
    let resetTimer;
    button.addEventListener("click", async () => {
      const code = pre.querySelector("code")?.textContent || pre.textContent || "";
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
        await Promise.race([
          navigator.clipboard.writeText(code.trim()),
          new Promise((_, reject) => window.setTimeout(
            () => reject(new Error("Clipboard write timed out")),
            750,
          )),
        ]);
        button.textContent = "Copied";
        button.setAttribute("aria-label", "Copied command");
        status.textContent = "Command copied to the clipboard.";
      } catch (_error) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(pre.querySelector("code") || pre);
        selection?.removeAllRanges();
        selection?.addRange(range);
        button.textContent = "Selected";
        button.setAttribute("aria-label", "Command selected; copy it manually");
        status.textContent = "Command selected. Press Command+C or Control+C to copy it.";
      }
      window.clearTimeout(resetTimer);
      resetTimer = window.setTimeout(() => {
        button.textContent = "Copy";
        button.setAttribute("aria-label", "Copy command");
      }, 2000);
    });
    pre.append(button, status);
  });

  const search = document.querySelector("#docs-search");
  const status = document.querySelector("#search-status");
  const noResults = document.querySelector("#no-results");
  const sections = [...document.querySelectorAll(".doc-section")];
  const navLinks = [...document.querySelectorAll(".docs-nav a[href^='#']")];

  const runSearch = () => {
    if (!(search instanceof HTMLInputElement)) return;
    const query = search.value.trim().toLocaleLowerCase();
    let matches = 0;
    sections.forEach((section) => {
      const haystack = `${section.dataset.search || ""} ${section.textContent || ""}`.toLocaleLowerCase();
      const hit = !query || haystack.includes(query);
      section.hidden = !hit;
      if (hit) matches += 1;
    });
    navLinks.forEach((link) => {
      const id = link.getAttribute("href")?.slice(1);
      const section = id ? document.getElementById(id) : null;
      link.closest("li")?.toggleAttribute("hidden", Boolean(section?.hidden));
    });
    if (status) {
      status.textContent = query ? `${matches} section${matches === 1 ? "" : "s"} matched` : "Search all documentation";
    }
    if (noResults) noResults.hidden = matches !== 0;
  };

  if (search instanceof HTMLInputElement) {
    search.addEventListener("input", runSearch);
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const typing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target?.isContentEditable;
      if (event.key === "/" && !typing) {
        event.preventDefault();
        search.focus();
      } else if (event.key === "Escape" && document.activeElement === search) {
        search.value = "";
        runSearch();
        search.blur();
      }
    });
    runSearch();
  }

  if ("IntersectionObserver" in window && navLinks.length) {
    const byId = new Map(navLinks.map((link) => [link.getAttribute("href")?.slice(1), link]));
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting && !entry.target.hidden)
        .sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top)[0];
      if (!visible) return;
      navLinks.forEach((link) => link.removeAttribute("aria-current"));
      byId.get(visible.target.id)?.setAttribute("aria-current", "true");
    }, { rootMargin: "-18% 0px -70%", threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }
})();
