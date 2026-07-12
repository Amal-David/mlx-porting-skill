(function attachOptimizationAtlas(globalScope, factory) {
  "use strict";

  const api = factory(globalScope);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (globalScope) globalScope.MLX_OPTIMIZATION_ATLAS = api;
  if (globalScope && typeof document !== "undefined") api.mountOptimizationAtlas();
}(typeof window !== "undefined" ? window : null, (globalScope) => {
  "use strict";

  const repository = "https://github.com/Amal-David/mlx-porting-skill";
  const familyOrder = [
    "evaluation-scheduling",
    "native-operators-compilation",
    "layout-numerics",
    "state-memory",
    "compression",
    "inference-algorithms",
    "serving-pipeline",
    "custom-backend",
  ];
  const pickerOrder = ["not-measured", ...familyOrder];
  const statusAdvisor = {
    "native-mlx": "validated-source-theory",
    "official-mlx-project": "validated-source-theory",
    "proven-mlx-port": "benchmark-required",
    "research-candidate": "experimental-approach",
    "rejected-or-superseded": "rejected-do-not-use",
  };
  const string = (value) => typeof value === "string" && value.trim().length > 0;
  const strings = (value) => Array.isArray(value) && value.length > 0 && value.every(string);
  const httpsUrl = (value) => {
    if (!string(value)) return false;
    try { return new URL(value).protocol === "https:"; } catch (_error) { return false; }
  };

  const statusKind = (status) => {
    if (status === "rejected-or-superseded") return "rejected";
    if (status === "research-candidate") return "experimental";
    if (status in statusAdvisor) return "implementation-backed";
    return "unavailable";
  };

  const selectionEligibility = (method, state) => {
    if (!state?.parityDeclared) return { allowed: false, reason: "parity-required" };
    if (!state?.familySelected) return { allowed: false, reason: "profile-required" };
    if (!state?.journeyCurated) return { allowed: false, reason: "journey-alternative" };
    const kind = statusKind(method?.status);
    if (kind === "rejected" || kind === "unavailable" || method?.advisor?.id === "rejected-do-not-use") {
      return { allowed: false, reason: "unavailable" };
    }
    if (kind === "experimental" && !state.researchOptIn) {
      return { allowed: false, reason: "research-opt-in-required" };
    }
    if (method?.advisor?.requires_user_opt_in && !state.researchOptIn) {
      return { allowed: false, reason: "advisor-opt-in-required" };
    }
    return { allowed: true, reason: "eligible-hypothesis" };
  };

  const validateOptimizationLearning = (learning) => {
    const families = learning?.optimization_families;
    const methods = learning?.guidance_methods;
    const journeys = learning?.journeys;
    if (!Array.isArray(families) || families.map((family) => family?.id).join("|") !== familyOrder.join("|")) {
      return false;
    }
    if (!Array.isArray(methods) || methods.length === 0) return false;
    const methodIds = methods.map((method) => method?.id);
    if (new Set(methodIds).size !== methodIds.length) return false;
    const categorized = families.flatMap((family) => family.method_ids || []);
    if (categorized.length !== methodIds.length || new Set(categorized).size !== categorized.length) return false;
    if (!methodIds.every((identifier) => categorized.includes(identifier))) return false;
    if (!Array.isArray(journeys) || journeys.length === 0) return false;
    if (!journeys.every((journey) => (
      ["id", "title", "status", "proof_boundary"].every((field) => string(journey?.[field]))
      && ["proven", "simulation"].includes(journey.status)
      && strings(journey.optimization_method_ids)
      && journey.optimization_method_ids.every((methodId) => methodIds.includes(methodId))
      && Array.isArray(journey.runbooks)
      && journey.runbooks.length > 0
      && journey.runbooks.every((runbook) => (
        string(runbook?.id)
        && string(runbook?.label)
        && /^references\/runbook-[a-z0-9-]+\.md$/.test(runbook?.path || "")
      ))
    ))) return false;
    if (!families.every((family) => (
      ["id", "title", "bottleneck", "proof_gate", "rollback"].every((field) => string(family?.[field]))
      && strings(family.method_ids)
    ))) return false;
    const familyByMethod = new Map(
      families.flatMap((family) => family.method_ids.map((methodId) => [methodId, family.id])),
    );
    return methods.every((method) => (
      [
        "id", "title", "status", "family_id", "recommendation", "expected_effect",
        "claim_eligibility", "numeric_authority", "prerequisite", "proof_gate",
      ].every((field) => string(method?.[field]))
      && strings(method.applies_to)
      && strings(method.tradeoffs)
      && strings(method.validation_gates)
      && strings(method.rollback_conditions)
      && Array.isArray(method.quality_gate)
      && typeof method.quality_gated === "boolean"
      && Array.isArray(method.evidence_links)
      && method.evidence_links.length > 0
      && method.evidence_links.every((link) => {
        if (!string(link?.title) || !string(link?.url)) return false;
        if (!string(link?.id) || !string(link?.role)
          || !["synthesized", "screened"].includes(link?.review_depth)
          || ![
            "context_only", "local_reproduced", "official_mlx", "official_mlx_project",
            "paper_only", "third_party_pinned", "unspecified",
          ].includes(link?.support_scope)
          || !Array.isArray(link?.claim_types)
          || !link.claim_types.every(string)) return false;
        return httpsUrl(link.url)
      })
      && string(method.canonical_source?.id)
      && httpsUrl(method.canonical_source?.url)
      && method.evidence_links.some((link) => (
        link.id === method.canonical_source.id
        && link.url === method.canonical_source.url
        && link.review_depth === method.canonical_source.review_depth
        && link.support_scope === method.canonical_source.support_scope
      ))
      && string(method.advisor?.id)
      && string(method.advisor?.label)
      && string(method.advisor?.description)
      && typeof method.advisor?.requires_user_opt_in === "boolean"
      && statusAdvisor[method.status] === method.advisor.id
      && familyByMethod.get(method.id) === method.family_id
      && (method.quality_gated ? method.quality_gate.length > 0 : method.quality_gate.length === 0)
      && method.numeric_authority === "effective_claims"
      && ((method.claim_eligibility === "local-promotion") === (method.numeric_claim !== null))
      && (
        method.numeric_claim === null
        || (
          method.claim_eligibility === "local-promotion"
          && string(method.numeric_claim?.range)
          && string(method.numeric_claim?.metric)
          && method.numeric_claim?.target_constraints
          && typeof method.numeric_claim.target_constraints === "object"
          && !Array.isArray(method.numeric_claim.target_constraints)
          && Object.keys(method.numeric_claim.target_constraints).length > 0
          && /^[a-f0-9]{64}$/.test(method.numeric_claim?.experiment_fingerprint || "")
        )
      )
    ));
  };

  const statusLabel = (method) => {
    const labels = {
      "native-mlx": "Native MLX path",
      "official-mlx-project": "Official MLX project",
      "proven-mlx-port": "Pinned MLX port evidence",
      "research-candidate": "Research candidate",
      "rejected-or-superseded": "Rejected / superseded",
    };
    return labels[method.status] || method.status;
  };

  const claimLabel = (method) => {
    if (method.numeric_claim) {
      return `Effective catalogued range: ${method.numeric_claim.range} (${method.numeric_claim.metric}); exact target constraints and experiment fingerprint apply`;
    }
    if (method.claim_eligibility === "withheld") return "Withheld — no number is eligible for guidance";
    return "No effective numeric claim";
  };

  const appendDefinition = (list, term, value) => {
    const row = document.createElement("div");
    const name = document.createElement("dt");
    const description = document.createElement("dd");
    name.textContent = term;
    description.textContent = value;
    row.append(name, description);
    list.append(row);
  };

  const exportExperimentPlan = (selected, methodById, journey = null) => {
    const lines = [
      "MLX optimization experiment hypotheses",
      "",
      "Planning boundary: these are hypotheses, not enabled methods, recommendations, or validated speed claims.",
    ];
    if (journey) {
      lines.push(`Journey context: ${journey.title} (${journey.status})`);
      lines.push(`Journey proof boundary: ${journey.proof_boundary}`);
      lines.push("Journey runbooks:");
      journey.runbooks.forEach((runbook) => {
        lines.push(`- ${runbook.label}: ${repository}/blob/main/mlx-model-porting/${runbook.path}`);
      });
    }
    if (selected.size === 0) {
      lines.push("", "No hypotheses selected.");
    } else {
      [...selected].forEach((methodId, index) => {
        const method = methodById.get(methodId);
        lines.push("", `${index + 1}. ${method.title} (${method.id})`);
        lines.push(`   Evidence: ${statusLabel(method)}; ${claimLabel(method)}`);
        lines.push(`   Prerequisite: ${method.prerequisite}`);
        lines.push(`   Proof gate: ${method.proof_gate}`);
        if (method.quality_gated) lines.push(`   Quality gate: ${method.quality_gate.join("; ")}`);
        lines.push(`   Validation: ${method.validation_gates.join("; ")}`);
        lines.push(`   Rollback: ${method.rollback_conditions.join("; ")}`);
        lines.push(`   Source: ${method.canonical_source.url}`);
      });
    }
    lines.push(
      "",
      "Tested repository handoff",
      "python3 mlx-model-porting/scripts/inspect_model.py MODEL_PATH --output inspection.json",
      "python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --target-profile target-profile.json --objective OBJECTIVE --output recommendations.json",
      "python3 mlx-model-porting/scripts/audit_skill.py --strict mlx-model-porting",
    );
    return lines.join("\n");
  };

  const mountOptimizationAtlas = () => {
    const learning = globalScope?.MLX_PORTING_SITE_DATA?.learning;
    const root = document.querySelector("[data-optimization-root]");
    const fallback = document.querySelector("[data-optimization-fallback]");
    if (!root || !fallback || !validateOptimizationLearning(learning)) return false;

    const radios = [...root.querySelectorAll('input[name="optimization-family"]')];
    if (radios.map((radio) => radio.value).join("|") !== pickerOrder.join("|")) return false;
    const parity = root.querySelector("[data-optimization-parity]");
    const research = root.querySelector("[data-optimization-research]");
    const lock = root.querySelector("[data-optimization-lock]");
    const title = root.querySelector("[data-optimization-family-title]");
    const bottleneck = root.querySelector("[data-optimization-bottleneck]");
    const journeyLabel = root.querySelector("[data-optimization-journey]");
    const proofGate = root.querySelector("[data-optimization-proof-gate]");
    const familyRollback = root.querySelector("[data-optimization-family-rollback]");
    const methodsRoot = root.querySelector("[data-optimization-methods]");
    const output = root.querySelector("[data-optimization-plan-output]");
    const copy = root.querySelector("[data-optimization-copy]");
    const live = root.querySelector("[data-optimization-live]");
    if (!parity || !research || !lock || !title || !bottleneck || !journeyLabel || !proofGate
      || !familyRollback || !methodsRoot || !output || !copy || !live) return false;

    const familyById = new Map(learning.optimization_families.map((family) => [family.id, family]));
    const methodById = new Map(learning.guidance_methods.map((method) => [method.id, method]));
    const selected = new Set();
    const openMethods = new Set();
    let familyId = null;
    const journeyById = new Map(learning.journeys.map((journey) => [journey.id, journey]));
    const journeyFromUrl = () => {
      try {
        const requested = new URL(globalScope.location.href).searchParams.get("atlas-model");
        return journeyById.get(requested) || learning.journeys[0];
      } catch (_error) {
        return learning.journeys[0];
      }
    };
    let journey = journeyFromUrl();

    const canSelect = (method) => selectionEligibility(method, {
      parityDeclared: parity.checked,
      familySelected: familyId !== null,
      researchOptIn: research.checked,
      journeyCurated: journey.optimization_method_ids.includes(method.id),
    }).allowed;

    const updateOutput = () => {
      output.value = exportExperimentPlan(selected, methodById, journey);
    };

    const renderMethods = (family) => {
      methodsRoot.replaceChildren();
      family.method_ids.forEach((methodId) => {
        const method = methodById.get(methodId);
        const kind = statusKind(method.status);
        const onJourney = journey.optimization_method_ids.includes(method.id);
        const details = document.createElement("details");
        details.className = `optimization-method ${kind} ${onJourney ? "on-journey" : "alternative"}`;
        details.open = openMethods.has(method.id);
        details.addEventListener("toggle", () => {
          if (details.open) openMethods.add(method.id);
          else openMethods.delete(method.id);
        });
        const summary = document.createElement("summary");
        const summaryText = document.createElement("span");
        const evidence = document.createElement("small");
        const heading = document.createElement("strong");
        evidence.textContent = statusLabel(method);
        heading.textContent = method.title;
        summaryText.append(evidence, heading);
        const identifier = document.createElement("code");
        identifier.textContent = method.id;
        summary.append(summaryText, identifier);
        const body = document.createElement("div");
        body.className = "optimization-method-body";
        const recommendation = document.createElement("p");
        recommendation.textContent = method.recommendation;
        const definitions = document.createElement("dl");
        appendDefinition(definitions, "Applies to", method.applies_to.join(", "));
        appendDefinition(
          definitions,
          "Journey applicability",
          onJourney
            ? `Curated teaching hypothesis for ${journey.title}; still requires exact-model validation.`
            : `Alternative branch outside the curated ${journey.title} journey; not selectable as model-specific guidance.`,
        );
        appendDefinition(definitions, "Prerequisite", method.prerequisite);
        appendDefinition(definitions, "Expected effect", method.expected_effect);
        appendDefinition(definitions, "Trade-offs", method.tradeoffs.join("; "));
        appendDefinition(definitions, "Evidence status", statusLabel(method));
        appendDefinition(definitions, "Advisor boundary", `${method.advisor.label}. ${method.advisor.description}`);
        appendDefinition(definitions, "Proof gate", method.proof_gate);
        if (method.quality_gated) {
          appendDefinition(definitions, "Quality gate (lossy)", method.quality_gate.join("; "));
        }
        appendDefinition(definitions, "Validation", method.validation_gates.join("; "));
        appendDefinition(definitions, "Rollback", method.rollback_conditions.join("; "));
        appendDefinition(definitions, "Numeric claim", claimLabel(method));
        if (method.numeric_claim) {
          appendDefinition(
            definitions,
            "Numeric scope",
            `${JSON.stringify(method.numeric_claim.target_constraints)}; fingerprint ${method.numeric_claim.experiment_fingerprint}`,
          );
        }
        const links = document.createElement("div");
        links.className = "optimization-evidence-links";
        method.evidence_links.slice(0, 3).forEach((link) => {
          const anchor = document.createElement("a");
          anchor.href = link.url;
          anchor.textContent = `${link.title} ↗`;
          anchor.rel = "noopener";
          links.append(anchor);
        });
        const add = document.createElement("button");
        add.type = "button";
        add.className = "button small optimization-add";
        add.dataset.optimizationAdd = method.id;
        add.disabled = !canSelect(method);
        add.setAttribute("aria-pressed", String(selected.has(method.id)));
        add.textContent = selected.has(method.id) ? "Remove hypothesis" : "Add hypothesis";
        if (kind === "rejected") add.textContent = "Unavailable — rejected";
        else if (!onJourney) add.textContent = "Unavailable — alternative journey branch";
        else if (!parity.checked) add.textContent = "Locked — parity required";
        else if (method.advisor.requires_user_opt_in && !research.checked) add.textContent = "Locked — research opt-in required";
        add.addEventListener("click", () => {
          if (!canSelect(method)) return;
          if (selected.has(method.id)) selected.delete(method.id);
          else selected.add(method.id);
          openMethods.add(method.id);
          render();
          const restored = methodsRoot.querySelector(`[data-optimization-add="${method.id}"]`);
          restored?.focus();
          live.textContent = `${method.title} ${selected.has(method.id) ? "added to" : "removed from"} the experiment plan.`;
        });
        body.append(recommendation, definitions, links, add);
        details.append(summary, body);
        methodsRoot.append(details);
      });
    };

    const render = () => {
      const family = familyId ? familyById.get(familyId) : null;
      journeyLabel.textContent = `Journey context: ${journey.title} · ${journey.status}`;
      if (!family) {
        lock.textContent = "Locked · measured bottleneck required";
        lock.className = "optimization-lock";
        title.textContent = "Select what profiling found";
        bottleneck.textContent = "Choose one measured branch before viewing or adding experiment hypotheses.";
        proofGate.textContent = "First select a measured bottleneck.";
        familyRollback.textContent = "Every selected experiment must retain the last parity-passing baseline.";
        methodsRoot.replaceChildren();
        updateOutput();
        return;
      }
      const unlocked = parity.checked;
      lock.textContent = unlocked
        ? "Planning unlocked · evidence not validated by this page"
        : "Locked · parity evidence required";
      lock.className = `optimization-lock ${unlocked ? "unlocked" : ""}`;
      title.textContent = family.title;
      bottleneck.textContent = family.bottleneck;
      proofGate.textContent = family.proof_gate;
      familyRollback.textContent = family.rollback;
      renderMethods(family);
      updateOutput();
    };

    radios.forEach((radio) => {
      radio.addEventListener("change", () => {
        if (!radio.checked) return;
        if (radio.value === "not-measured") familyId = null;
        else if (familyById.has(radio.value)) familyId = radio.value;
        else return;
        selected.clear();
        openMethods.clear();
        render();
        live.textContent = familyId
          ? `${familyById.get(familyId).title} methods shown.`
          : "No measured bottleneck selected. Experiment planning remains locked.";
      });
    });
    parity.addEventListener("change", () => {
      if (!parity.checked) selected.clear();
      render();
      live.textContent = parity.checked && familyId
        ? "Experiment planning unlocked. Evidence has not been validated by this page."
        : parity.checked
          ? "Parity declared, but planning remains locked until a measured bottleneck is selected."
        : "Experiment planning locked and selected hypotheses cleared.";
    });
    research.addEventListener("change", () => {
      if (!research.checked) {
        [...selected].forEach((methodId) => {
          if (methodById.get(methodId).advisor.requires_user_opt_in) selected.delete(methodId);
        });
      }
      render();
      live.textContent = research.checked
        ? "Research hypotheses may now be selected."
        : "Research hypotheses locked and removed from the plan.";
    });
    globalScope.addEventListener("mlx-atlas-state-change", (event) => {
      journey = journeyById.get(event.detail?.journeyId) || journeyFromUrl();
      selected.clear();
      openMethods.clear();
      render();
      live.textContent = `${journey.title} journey context loaded; selected hypotheses cleared.`;
    });
    copy.addEventListener("click", async () => {
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
        await navigator.clipboard.writeText(output.value);
        copy.textContent = "Copied";
        live.textContent = "Experiment plan copied to the clipboard.";
      } catch (_error) {
        output.focus();
        output.select();
        copy.textContent = "Selected";
        live.textContent = "Experiment plan selected. Press Command+C or Control+C to copy it.";
      }
      globalScope.setTimeout(() => { copy.textContent = "Copy experiment plan"; }, 1800);
    });

    try {
      render();
    } catch (_error) {
      root.hidden = true;
      delete root.dataset.optimizationEnhanced;
      fallback.hidden = false;
      return false;
    }
    root.hidden = false;
    root.dataset.optimizationEnhanced = "true";
    fallback.hidden = true;
    return true;
  };

  return {
    validateOptimizationLearning,
    statusKind,
    selectionEligibility,
    exportExperimentPlan,
    mountOptimizationAtlas,
  };
}));
