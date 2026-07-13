(function attachAtlas(globalScope, factory) {
  "use strict";

  const api = factory(globalScope);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (globalScope) globalScope.MLX_PORTING_ATLAS = api;
  if (globalScope && typeof document !== "undefined") api.mountAtlas();
}(typeof window !== "undefined" ? window : null, (globalScope) => {
  "use strict";

  const repository = "https://github.com/Amal-David/mlx-porting-skill";
  const validArray = (value) => Array.isArray(value) && value.length > 0;
  const nonEmptyString = (value) => typeof value === "string" && value.trim().length > 0;
  const nodeFields = [
    "id", "title", "outcome", "concept", "why_mlx_differs", "inspect",
    "prerequisite", "proof", "evidence_state",
  ];

  const validateAtlasLearning = (learning) => {
    if (!learning || typeof learning !== "object") return false;
    const order = learning.checkpoint_order;
    const nodes = learning.checkpoint_nodes;
    const journeys = learning.journeys;
    if (!Array.isArray(order) || order.length !== 8 || new Set(order).size !== order.length) return false;
    if (!order.every(nonEmptyString)) return false;
    if (!Array.isArray(nodes) || nodes.length !== order.length) return false;
    if (!nodes.every((node) => nodeFields.every((field) => nonEmptyString(node?.[field])))) return false;
    if (nodes.map((node) => node.id).join("|") !== order.join("|")) return false;
    if (!validArray(journeys) || new Set(journeys.map((journey) => journey?.id)).size !== journeys.length) {
      return false;
    }
    return journeys.every((journey) => (
      ["id", "title", "status", "modality", "source_format", "proof_boundary"].every(
        (field) => nonEmptyString(journey?.[field]),
      )
      && validArray(journey.architecture_ids)
      && journey.architecture_ids.every(nonEmptyString)
      && validArray(journey.runbooks)
      && journey.runbooks.every((runbook) => nonEmptyString(runbook?.label) && nonEmptyString(runbook?.path))
      && validArray(journey.component_path)
      && journey.component_path.every((component) => nonEmptyString(component?.title))
      && journey.checkpoint_notes
      && typeof journey.checkpoint_notes === "object"
      && order.every((checkpointId) => nonEmptyString(journey.checkpoint_notes[checkpointId]))
    ));
  };

  const journeyIds = (learning) => (
    validArray(learning?.journeys) ? learning.journeys.map((journey) => journey.id) : []
  );

  const checkpointIds = (learning) => (
    validArray(learning?.checkpoint_order) ? Array.from(learning.checkpoint_order) : []
  );

  const parseAtlasState = (learning, value) => {
    const journeys = journeyIds(learning);
    const checkpoints = checkpointIds(learning);
    const fallbackJourney = journeys[0] || "qwen25-dense-decoder";
    const fallbackCheckpoint = checkpoints[0] || "inspect";
    let url;
    try {
      url = new URL(value, "https://example.invalid/");
    } catch (_error) {
      return { journeyId: fallbackJourney, checkpointId: fallbackCheckpoint, pathOnly: true };
    }
    const requestedJourney = url.searchParams.get("atlas-model");
    const requestedCheckpoint = url.searchParams.get("atlas-node");
    return {
      journeyId: journeys.includes(requestedJourney) ? requestedJourney : fallbackJourney,
      checkpointId: checkpoints.includes(requestedCheckpoint) ? requestedCheckpoint : fallbackCheckpoint,
      pathOnly: url.searchParams.get("atlas-path") !== "all",
    };
  };

  const serializeAtlasState = (base, state) => {
    const url = new URL(base, "https://example.invalid/");
    url.searchParams.set("atlas-model", String(state.journeyId));
    url.searchParams.set("atlas-node", String(state.checkpointId));
    if (state.pathOnly === false) url.searchParams.set("atlas-path", "all");
    else url.searchParams.delete("atlas-path");
    url.hash = "porting-atlas";
    return url.toString();
  };

  const moveCheckpointFocus = (order, current, key) => {
    if (!Array.isArray(order) || order.length === 0) return current;
    const index = Math.max(0, order.indexOf(current));
    if (key === "Home") return order[0];
    if (key === "End") return order[order.length - 1];
    if (key === "ArrowRight" || key === "ArrowDown") {
      return order[Math.min(order.length - 1, index + 1)];
    }
    if (key === "ArrowLeft" || key === "ArrowUp") {
      return order[Math.max(0, index - 1)];
    }
    return order[index];
  };

  const stepCheckpoint = (order, current, direction) => {
    if (!Array.isArray(order) || order.length === 0) return current;
    const index = Math.max(0, order.indexOf(current));
    const delta = direction === "previous" ? -1 : direction === "next" ? 1 : 0;
    return order[Math.max(0, Math.min(order.length - 1, index + delta))];
  };

  const exportTextPlan = (learning, state) => {
    const journey = learning?.journeys?.find((candidate) => candidate.id === state.journeyId)
      || learning?.journeys?.[0];
    if (!journey) return "MLX port plan\n\nNo canonical journey data is available.";
    const nodes = new Map((learning.checkpoint_nodes || []).map((node) => [node.id, node]));
    const statusDescription = learning.journey_statuses?.[journey.status] || journey.status;
    const components = (journey.component_path || []).map((component) => component.title).join(" -> ");
    const lines = [
      `MLX port plan — ${journey.title}`,
      "",
      `Journey status: ${journey.status}`,
      `Evidence status: ${statusDescription}`,
      `Exact proof boundary: ${journey.proof_boundary}`,
      `Modality: ${journey.modality}`,
      `Source format: ${journey.source_format}`,
      `Architecture routes: ${(journey.architecture_ids || []).join(", ")}`,
      `Component path: ${components}`,
      `Selected checkpoint: ${nodes.get(state.checkpointId)?.title || state.checkpointId}`,
      "",
      "Eight checkpoint rail",
    ];
    (learning.checkpoint_order || []).forEach((checkpointId, index) => {
      const title = nodes.get(checkpointId)?.title || checkpointId;
      const note = journey.checkpoint_notes?.[checkpointId] || "No canonical note recorded.";
      lines.push(`${index + 1}. ${title}`);
      lines.push(`   ${note}`);
    });
    lines.push("");
    lines.push("Architecture runbooks");
    (journey.runbooks || []).forEach((runbook) => {
      lines.push(`- ${runbook.label}: ${repository}/blob/main/mlx-model-porting/${runbook.path}`);
    });
    lines.push("");
    lines.push("Tested repository handoff");
    lines.push("python3 mlx-model-porting/scripts/inspect_model.py MODEL_PATH --output inspection.json");
    lines.push("python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --target-profile target-profile.json --objective OBJECTIVE --output recommendations.json");
    lines.push("python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting");
    lines.push("");
    lines.push("Selection records a study position only. It does not mark implementation, parity, or validation complete.");
    return lines.join("\n");
  };

  const shortStatus = (journey) => (
    journey.status === "proven"
      ? "Proven — pinned checkpoint proof"
      : "Simulation — not a completed checkpoint port"
  );

  const mountAtlas = () => {
    const data = globalScope?.MLX_PORTING_SITE_DATA;
    const learning = data?.learning;
    const root = document.querySelector("[data-atlas-root]");
    const fallback = document.querySelector("[data-atlas-fallback]");
    if (!root || !fallback || !validateAtlasLearning(learning)) return false;

    const journeys = learning.journeys;
    const order = checkpointIds(learning);
    const nodes = learning.checkpoint_nodes;
    const radios = [...root.querySelectorAll('input[name="atlas-journey"]')];
    const nodeButtons = [...root.querySelectorAll("[data-atlas-node]")];
    const expectedJourneys = journeyIds(learning);
    if (
      !validArray(journeys)
      || order.length !== 8
      || nodes?.length !== order.length
      || radios.map((radio) => radio.value).join("|") !== expectedJourneys.join("|")
      || nodeButtons.map((button) => button.dataset.atlasNode).join("|") !== order.join("|")
    ) return false;

    const journeyById = new Map(journeys.map((journey) => [journey.id, journey]));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const familyById = new Map(
      (data.architectures?.families || []).map((family) => [family.id, family]),
    );
    const live = root.querySelector("[data-atlas-live]");
    const status = root.querySelector("[data-atlas-status]");
    const journeyTitle = root.querySelector("[data-atlas-journey-title]");
    const componentPath = root.querySelector("[data-atlas-component-path]");
    const stepLabel = root.querySelector("[data-atlas-step-label]");
    const stepTitle = root.querySelector("[data-atlas-step-title]");
    const journeyNote = root.querySelector("[data-atlas-journey-note]");
    const proofBoundary = root.querySelector("[data-atlas-proof-boundary]");
    const runbooks = root.querySelector("[data-atlas-runbooks]");
    const comparison = root.querySelector("[data-atlas-comparison]");
    const comparisonList = root.querySelector("[data-atlas-comparison-list]");
    const pathOnly = root.querySelector("[data-atlas-path-only]");
    const previous = root.querySelector("[data-atlas-previous]");
    const next = root.querySelector("[data-atlas-next]");
    const output = root.querySelector("[data-atlas-export-output]");
    const copy = root.querySelector("[data-atlas-copy]");
    const fieldNodes = new Map(
      [...root.querySelectorAll("[data-atlas-field]")].map((node) => [node.dataset.atlasField, node]),
    );
    if (
      !live || !status || !journeyTitle || !componentPath || !stepLabel || !stepTitle
      || !proofBoundary
      || !journeyNote || !runbooks || !comparison || !comparisonList || !pathOnly
      || !previous || !next || !output || !copy || fieldNodes.size < 6
    ) return false;

    let state = parseAtlasState(learning, globalScope.location.href);
    let copyResetTimer;

    const writeUrl = (replace = false) => {
      const value = serializeAtlasState(globalScope.location.href, state);
      const method = replace ? "replaceState" : "pushState";
      globalScope.history[method]({ atlas: state }, "", value);
      globalScope.dispatchEvent(new CustomEvent("mlx-atlas-state-change", { detail: state }));
    };

    const announce = (journey, node) => {
      const position = order.indexOf(node.id) + 1;
      live.textContent = `${journey.title}, ${shortStatus(journey)}. Step ${position} of ${order.length}, ${node.title}.`;
    };

    const renderRunbooks = (journey) => {
      runbooks.replaceChildren();
      journey.architecture_ids.forEach((architectureId) => {
        const family = familyById.get(architectureId);
        if (!family?.runbook) return;
        const link = document.createElement("a");
        link.href = `${repository}/blob/main/mlx-model-porting/${family.runbook}`;
        link.textContent = `${family.label} runbook ↗`;
        link.rel = "noopener";
        runbooks.append(link);
      });
    };

    const renderComparison = (checkpointId) => {
      comparison.hidden = state.pathOnly;
      comparisonList.replaceChildren();
      if (state.pathOnly) return;
      journeys.forEach((journey) => {
        const item = document.createElement("li");
        const heading = document.createElement("strong");
        heading.textContent = `${journey.title} — ${shortStatus(journey)}`;
        const note = document.createElement("span");
        note.textContent = journey.checkpoint_notes[checkpointId];
        item.append(heading, document.createElement("br"), note);
        comparisonList.append(item);
      });
    };

    const render = ({ shouldAnnounce = false } = {}) => {
      const journey = journeyById.get(state.journeyId) || journeys[0];
      const node = nodeById.get(state.checkpointId) || nodes[0];
      const position = order.indexOf(node.id);
      radios.forEach((radio) => { radio.checked = radio.value === journey.id; });
      nodeButtons.forEach((button) => {
        const selected = button.dataset.atlasNode === node.id;
        button.setAttribute("aria-pressed", String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
      status.textContent = shortStatus(journey);
      status.className = `atlas-status ${journey.status}`;
      journeyTitle.textContent = journey.title;
      componentPath.textContent = journey.component_path.map((component) => component.title).join(" → ");
      stepLabel.textContent = `Checkpoint ${String(position + 1).padStart(2, "0")} / ${String(order.length).padStart(2, "0")}`;
      stepTitle.textContent = node.title;
      journeyNote.textContent = journey.checkpoint_notes[node.id];
      proofBoundary.textContent = journey.proof_boundary;
      ["concept", "why_mlx_differs", "inspect", "proof", "prerequisite"].forEach((field) => {
        fieldNodes.get(field).textContent = node[field];
      });
      fieldNodes.get("evidence_state").textContent = `${node.evidence_state}. ${learning.journey_statuses[journey.status]}`;
      renderRunbooks(journey);
      renderComparison(node.id);
      pathOnly.checked = state.pathOnly;
      previous.disabled = position === 0;
      next.disabled = position === order.length - 1;
      output.value = exportTextPlan(learning, state);
      if (shouldAnnounce) announce(journey, node);
    };

    const selectCheckpoint = (checkpointId, { push = true, shouldAnnounce = true } = {}) => {
      if (!order.includes(checkpointId)) return;
      state = { ...state, checkpointId };
      render({ shouldAnnounce });
      if (push) writeUrl();
    };

    radios.forEach((radio) => {
      radio.addEventListener("change", () => {
        if (!radio.checked || !journeyById.has(radio.value)) return;
        state = { ...state, journeyId: radio.value };
        render({ shouldAnnounce: true });
        writeUrl();
      });
    });

    nodeButtons.forEach((button) => {
      button.addEventListener("click", () => selectCheckpoint(button.dataset.atlasNode));
      button.addEventListener("keydown", (event) => {
        const navigationKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"];
        if (!navigationKeys.includes(event.key)) return;
        const targetId = moveCheckpointFocus(order, button.dataset.atlasNode, event.key);
        event.preventDefault();
        if (targetId === button.dataset.atlasNode) return;
        nodeButtons.forEach((candidate) => {
          candidate.tabIndex = candidate.dataset.atlasNode === targetId ? 0 : -1;
        });
        nodeButtons.find((candidate) => candidate.dataset.atlasNode === targetId)?.focus();
      });
    });

    previous.addEventListener("click", () => {
      selectCheckpoint(stepCheckpoint(order, state.checkpointId, "previous"));
    });
    next.addEventListener("click", () => {
      selectCheckpoint(stepCheckpoint(order, state.checkpointId, "next"));
    });
    pathOnly.addEventListener("change", () => {
      state = { ...state, pathOnly: pathOnly.checked };
      render({ shouldAnnounce: true });
      writeUrl();
    });
    copy.addEventListener("click", async () => {
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
        await navigator.clipboard.writeText(output.value);
        live.textContent = "Port plan copied to the clipboard.";
        copy.textContent = "Copied";
      } catch (_error) {
        output.focus();
        output.select();
        live.textContent = "Port plan selected. Press Command+C or Control+C to copy it.";
        copy.textContent = "Selected";
      }
      globalScope.clearTimeout(copyResetTimer);
      copyResetTimer = globalScope.setTimeout(() => { copy.textContent = "Copy plan"; }, 1800);
    });
    globalScope.addEventListener("popstate", () => {
      state = parseAtlasState(learning, globalScope.location.href);
      render({ shouldAnnounce: true });
      globalScope.dispatchEvent(new CustomEvent("mlx-atlas-state-change", { detail: state }));
    });

    try {
      render();
    } catch (_error) {
      root.hidden = true;
      delete root.dataset.atlasEnhanced;
      fallback.hidden = false;
      return false;
    }
    root.hidden = false;
    root.dataset.atlasEnhanced = "true";
    fallback.hidden = true;
    return true;
  };

  return {
    validateAtlasLearning,
    parseAtlasState,
    serializeAtlasState,
    moveCheckpointFocus,
    stepCheckpoint,
    exportTextPlan,
    mountAtlas,
  };
}));
