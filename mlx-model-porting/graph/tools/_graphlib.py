"""Shared strict-JSON loading and evidence-graph validation primitives.

Standard library only. Every consumer of the v2 evidence graph in this
repository goes through this module so that one definition of "valid" is
enforced by the validator, the compiler, the pack gate, and the tests.

The upstream contract is `graph/schema/evidence-graph.schema.json`, copied
from the auto-mlx tool repository. This module does not interpret arbitrary
JSON Schema; it hard-codes the constraints that matter and cross-checks the
enumerations against the schema file at load time so that the two cannot
drift silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_VERSION = 2

REPO_GRAPH_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_GRAPH_ROOT / "schema" / "evidence-graph.schema.json"
SHARDS_DIR = REPO_GRAPH_ROOT / "shards"
COMPILED_PATH = REPO_GRAPH_ROOT / "compiled" / "evidence-graph.json"
PACKS_DIR = REPO_GRAPH_ROOT / "packs"

NODE_KINDS = (
    "model",
    "trait",
    "hardware",
    "workload",
    "mechanism",
    "applied_result",
    "hypothesis",
    "constraint",
    "finding",
    "external_reference",
    "frontier",
)

RELATIONS = (
    "contains",
    "supports",
    "constrains",
    "depends_on",
    "overlaps",
    "contradicts",
    "invalidates",
    "suggests",
    "requires_gate",
    "supersedes",
    "cannot_validate",
    "competes_with",
    "duplicates",
    "exhibits",
    "applies_to",
    "conditioned_on",
    "instantiates",
    "applied_on",
    "measured_on",
    "under_workload",
    "transfer_predicted",
    "confirms",
    "refutes",
)

CONFIDENCE = ("high", "medium", "low")

PROVENANCE = (
    "official_verified",
    "code_verified",
    "local_measured",
    "author_claim",
    "contributor_claim",
    "replicated",
    "research_inference",
    "transfer_inference",
)

EXACTNESS_CLASSES = (
    "exact_by_construction",
    "needs_parity_proof",
    "approximate_legal",
)

VERDICTS = ("improved", "regressed", "inconclusive")

NODE_REQUIRED = (
    "id",
    "kind",
    "title",
    "status",
    "confidence",
    "provenance",
    "summary",
    "evidence",
    "tags",
    "observed_at",
)
NODE_OPTIONAL = ("identity", "exactness_class", "effect")

EDGE_REQUIRED = ("from", "to", "relation", "confidence", "rationale")
EDGE_OPTIONAL = ("observed_at",)

EFFECT_REQUIRED = ("metric", "verdict", "delta_bp_ci")
EFFECT_OPTIONAL = ("receipt_id", "baseline_id", "sample_note")

DOC_REQUIRED = ("schema_version", "graph_id", "generated_at", "nodes", "edges")

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._/@-]*:[A-Za-z0-9][A-Za-z0-9._/@+-]*$")
RECEIPT_RE = re.compile(r"^[0-9a-f]{64}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T[0-9:.+Z-]+)?$")

ID_MIN_LENGTH = 3
ID_MAX_LENGTH = 300

# These caps are enforced by the auto-mlx executable validator but are NOT
# expressed in the copied JSON Schema, so `check_schema_agreement` cannot see
# drift in them. Mirror them here so a document that passes locally also passes
# the tool.
MAX_TAGS = 64
MAX_EVIDENCE = 64
MAX_IDENTITY_PROPERTIES = 32

# Node id prefix must name the node kind. `applied_result` uses the shorter
# `result:` prefix and `external_reference` uses `external:` because those are
# the forms already used by the upstream campaign graph.
KIND_PREFIX = {
    "model": "model",
    "trait": "trait",
    "hardware": "hardware",
    "workload": "workload",
    "mechanism": "mechanism",
    "applied_result": "result",
    "hypothesis": "hypothesis",
    "constraint": "constraint",
    "finding": "finding",
    "external_reference": "external",
    "frontier": "frontier",
}

# Typed endpoints. Relations absent from this table accept any node kinds.
ENDPOINT_RULES = {
    "exhibits": ({"model"}, {"trait"}),
    "applies_to": ({"mechanism"}, {"trait"}),
    "instantiates": ({"applied_result"}, {"mechanism"}),
    "applied_on": ({"applied_result"}, {"model"}),
    "measured_on": ({"applied_result"}, {"hardware"}),
    "under_workload": ({"applied_result"}, {"workload"}),
    "transfer_predicted": ({"mechanism"}, {"model"}),
    "confirms": ({"applied_result"}, {"mechanism", "hypothesis"}),
    "refutes": ({"applied_result"}, {"mechanism", "hypothesis"}),
}

# Exactly one of each of these must leave every applied_result node; the
# reified hyperedge is not a hyperedge without all three anchors.
RESULT_EXACTLY_ONE = ("applied_on", "measured_on", "under_workload")
RESULT_AT_LEAST_ONE = ("instantiates",)

# Any measurement claim that is not officially or locally verified must not be
# recorded with a provenance stronger than a claim.
MEASUREMENT_PROVENANCE = frozenset(
    {"official_verified", "local_measured", "replicated", "code_verified"}
)

PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'(\[<])(?:/Users/|/home/|/private/(?:tmp|var)/|/var/folders/"
    r"|[A-Za-z]:\\\\Users\\\\)",
)

SECRET_PATTERNS = (
    ("anthropic-api-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer-header", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    (
        "inline-credential",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token)\b"
            r"\s*[:=]\s*[\"'][^\"']{12,}[\"']"
        ),
    ),
    ("url-userinfo", re.compile(r"://[^/\s:@]+:[^/\s@]+@")),
)


class GraphError(Exception):
    """Raised when a document cannot be parsed under strict-JSON discipline."""


def _reject_float(raw):
    raise GraphError(
        "float literal %r is forbidden; the graph is strict-integer JSON "
        "(effects are integer basis points)" % raw
    )


def _reject_constant(name):
    raise GraphError("JSON constant %r is forbidden" % name)


def _reject_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise GraphError("duplicate object key %r" % key)
        seen[key] = value
    return seen


def load_strict(path):
    """Parse JSON with no floats, no NaN/Infinity, and no duplicate keys."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except GraphError as exc:
        raise GraphError("%s: %s" % (path, exc)) from exc
    except ValueError as exc:
        raise GraphError("%s: invalid JSON: %s" % (path, exc)) from exc


def dump_canonical(obj):
    """Deterministic serialization: sorted keys, LF endings, UTF-8, no floats."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_canonical(path, obj):
    Path(path).write_text(dump_canonical(obj), encoding="utf-8", newline="\n")


def _is_bool(value):
    return isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _str_problem(value, field, minlen=1, maxlen=None):
    if not isinstance(value, str):
        return "%s must be a string" % field
    if len(value) < minlen:
        return "%s must be at least %d characters" % (field, minlen)
    if maxlen is not None and len(value) > maxlen:
        return "%s must be at most %d characters (got %d)" % (field, maxlen, len(value))
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return "%s contains an unpaired surrogate" % field
    return None


def iter_strings(obj, path="$"):
    """Yield (json_pointer_ish_path, string) for every string in a document."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield path + "." + str(key), str(key)
            yield from iter_strings(value, path + "." + str(key))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from iter_strings(value, "%s[%d]" % (path, index))


def scan_secrets(obj, where=""):
    """Return findings for credential-shaped strings anywhere in a document."""
    findings = []
    for path, text in iter_strings(obj, where or "$"):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append("%s: possible %s" % (path, name))
    return findings


def scan_private_paths(obj, where=""):
    """Return findings for absolute local filesystem paths in a document."""
    findings = []
    for path, text in iter_strings(obj, where or "$"):
        if PRIVATE_PATH_RE.search(text):
            findings.append(
                "%s: absolute private path; use a URL or a descriptive locator" % path
            )
    return findings


def check_schema_agreement():
    """Fail loudly if the copied schema no longer matches the hard-coded rules."""
    problems = []
    if not SCHEMA_PATH.exists():
        return ["schema file missing at %s" % SCHEMA_PATH]
    schema = load_strict(SCHEMA_PATH)
    defs = schema.get("$defs", {})
    pairs = (
        ("node kinds", tuple(defs["node"]["properties"]["kind"]["enum"]), NODE_KINDS),
        ("relations", tuple(defs["edge"]["properties"]["relation"]["enum"]), RELATIONS),
        ("confidence", tuple(defs["confidence"]["enum"]), CONFIDENCE),
        ("provenance", tuple(defs["provenance"]["enum"]), PROVENANCE),
        (
            "exactness_class",
            tuple(defs["node"]["properties"]["exactness_class"]["enum"]),
            EXACTNESS_CLASSES,
        ),
        ("verdict", tuple(defs["effect"]["properties"]["verdict"]["enum"]), VERDICTS),
        ("node required", tuple(defs["node"]["required"]), NODE_REQUIRED),
        ("edge required", tuple(defs["edge"]["required"]), EDGE_REQUIRED),
        ("effect required", tuple(defs["effect"]["required"]), EFFECT_REQUIRED),
        ("document required", tuple(schema["required"]), DOC_REQUIRED),
    )
    for label, upstream, local in pairs:
        if upstream != local:
            problems.append(
                "schema drift in %s: upstream %r != validator %r" % (label, upstream, local)
            )
    if defs["identifier"]["pattern"] != IDENTIFIER_RE.pattern:
        problems.append(
            "schema drift in identifier pattern: upstream %r != validator %r"
            % (defs["identifier"]["pattern"], IDENTIFIER_RE.pattern)
        )
    if schema.get("properties", {}).get("schema_version", {}).get("const") != SCHEMA_VERSION:
        problems.append("schema drift: schema_version const is not %d" % SCHEMA_VERSION)
    return problems


def validate_effect(effect, where):
    problems = []
    if not isinstance(effect, dict):
        return ["%s: effect must be an object" % where]
    unknown = set(effect) - set(EFFECT_REQUIRED) - set(EFFECT_OPTIONAL)
    if unknown:
        problems.append("%s: effect has unknown fields %s" % (where, sorted(unknown)))
    for field in EFFECT_REQUIRED:
        if field not in effect:
            problems.append("%s: effect is missing %r" % (where, field))
    if "metric" in effect:
        problem = _str_problem(effect["metric"], "effect.metric", 1, 100)
        if problem:
            problems.append("%s: %s" % (where, problem))
    if "verdict" in effect and effect["verdict"] not in VERDICTS:
        problems.append("%s: effect.verdict %r is not one of %s" % (where, effect["verdict"], list(VERDICTS)))
    interval = effect.get("delta_bp_ci")
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            problems.append("%s: effect.delta_bp_ci must be a two-item array" % where)
        elif not all(_is_int(value) for value in interval):
            problems.append(
                "%s: effect.delta_bp_ci must hold integers (basis points, never floats)" % where
            )
        else:
            low, high = interval
            if low > high:
                problems.append("%s: effect.delta_bp_ci is inverted (%d > %d)" % (where, low, high))
            for value in interval:
                if not -1000000 <= value <= 1000000:
                    problems.append("%s: effect.delta_bp_ci value %d is out of range" % (where, value))
    if "receipt_id" in effect and not (
        isinstance(effect["receipt_id"], str) and RECEIPT_RE.match(effect["receipt_id"])
    ):
        problems.append("%s: effect.receipt_id must be 64 lowercase hex characters" % where)
    for field, maxlen in (("baseline_id", 300), ("sample_note", 1000)):
        if field in effect:
            problem = _str_problem(effect[field], "effect." + field, 1, maxlen)
            if problem:
                problems.append("%s: %s" % (where, problem))
    return problems


def validate_node(node, where):
    problems = []
    if not isinstance(node, dict):
        return ["%s: node must be an object" % where]
    unknown = set(node) - set(NODE_REQUIRED) - set(NODE_OPTIONAL)
    if unknown:
        problems.append("%s: node has unknown fields %s" % (where, sorted(unknown)))
    for field in NODE_REQUIRED:
        if field not in node:
            problems.append("%s: node is missing required field %r" % (where, field))
    node_id = node.get("id")
    if isinstance(node_id, str):
        where = "%s (%s)" % (where, node_id)
        if not IDENTIFIER_RE.match(node_id):
            problems.append("%s: id does not match the prefix:slug identifier pattern" % where)
        if not ID_MIN_LENGTH <= len(node_id) <= ID_MAX_LENGTH:
            problems.append(
                "%s: id length %d is outside %d..%d"
                % (where, len(node_id), ID_MIN_LENGTH, ID_MAX_LENGTH)
            )
    elif "id" in node:
        problems.append("%s: id must be a string" % where)

    kind = node.get("kind")
    if kind not in NODE_KINDS:
        problems.append("%s: kind %r is not one of %s" % (where, kind, list(NODE_KINDS)))
    elif isinstance(node_id, str) and ":" in node_id:
        expected = KIND_PREFIX[kind]
        actual = node_id.split(":", 1)[0]
        if actual != expected:
            problems.append(
                "%s: kind %r requires the id prefix %r but the id uses %r"
                % (where, kind, expected, actual)
            )

    for field, maxlen in (("title", 300), ("status", 100), ("summary", 4000)):
        if field in node:
            problem = _str_problem(node[field], field, 1, maxlen)
            if problem:
                problems.append("%s: %s" % (where, problem))
    if node.get("confidence") not in CONFIDENCE:
        problems.append("%s: confidence %r is not one of %s" % (where, node.get("confidence"), list(CONFIDENCE)))
    if node.get("provenance") not in PROVENANCE:
        problems.append("%s: provenance %r is not one of %s" % (where, node.get("provenance"), list(PROVENANCE)))

    evidence = node.get("evidence")
    if not isinstance(evidence, list):
        problems.append("%s: evidence must be an array" % where)
    else:
        if not evidence:
            problems.append("%s: evidence is empty; every claim needs a locator" % where)
        if len(evidence) > MAX_EVIDENCE:
            problems.append(
                "%s: evidence has %d entries; the maximum is %d"
                % (where, len(evidence), MAX_EVIDENCE)
            )
        for index, item in enumerate(evidence):
            problem = _str_problem(item, "evidence[%d]" % index, 1, 1000)
            if problem:
                problems.append("%s: %s" % (where, problem))

    tags = node.get("tags")
    if not isinstance(tags, list):
        problems.append("%s: tags must be an array" % where)
    else:
        for index, item in enumerate(tags):
            problem = _str_problem(item, "tags[%d]" % index, 1, 100)
            if problem:
                problems.append("%s: %s" % (where, problem))
        if len(set(map(repr, tags))) != len(tags):
            problems.append("%s: tags must be unique" % where)
        if len(tags) > MAX_TAGS:
            problems.append(
                "%s: tags has %d entries; the maximum is %d" % (where, len(tags), MAX_TAGS)
            )

    observed_at = node.get("observed_at")
    problem = _str_problem(observed_at, "observed_at", 1, 64)
    if problem:
        problems.append("%s: %s" % (where, problem))
    elif not ISO_DATE_RE.match(observed_at):
        problems.append("%s: observed_at %r is not an ISO date" % (where, observed_at))

    identity = node.get("identity")
    if identity is not None:
        if not isinstance(identity, dict):
            problems.append("%s: identity must be an object" % where)
        else:
            if len(identity) > MAX_IDENTITY_PROPERTIES:
                problems.append(
                    "%s: identity has %d properties; the maximum is %d"
                    % (where, len(identity), MAX_IDENTITY_PROPERTIES)
                )
            for key, value in identity.items():
                if not (isinstance(value, str) or _is_int(value) or _is_bool(value)):
                    problems.append(
                        "%s: identity[%r] must be a string, integer, or boolean" % (where, key)
                    )

    if "exactness_class" in node:
        if node["exactness_class"] not in EXACTNESS_CLASSES:
            problems.append(
                "%s: exactness_class %r is not one of %s"
                % (where, node["exactness_class"], list(EXACTNESS_CLASSES))
            )
        if kind != "mechanism":
            problems.append("%s: exactness_class is only meaningful on mechanism nodes" % where)

    if "effect" in node:
        problems.extend(validate_effect(node["effect"], where))
        if kind != "applied_result":
            problems.append(
                "%s: effect may only appear on applied_result nodes; a %s node cannot carry a measurement"
                % (where, kind)
            )
    elif kind == "applied_result":
        problems.append("%s: applied_result nodes must carry an effect" % where)

    return problems


def validate_edge(edge, where):
    problems = []
    if not isinstance(edge, dict):
        return ["%s: edge must be an object" % where]
    unknown = set(edge) - set(EDGE_REQUIRED) - set(EDGE_OPTIONAL)
    if unknown:
        problems.append("%s: edge has unknown fields %s" % (where, sorted(unknown)))
    for field in EDGE_REQUIRED:
        if field not in edge:
            problems.append("%s: edge is missing required field %r" % (where, field))
    for field in ("from", "to"):
        value = edge.get(field)
        if not isinstance(value, str):
            problems.append("%s: edge.%s must be a string" % (where, field))
        elif not IDENTIFIER_RE.match(value):
            problems.append("%s: edge.%s %r is not a valid identifier" % (where, field, value))
    if edge.get("relation") not in RELATIONS:
        problems.append("%s: relation %r is not one of %s" % (where, edge.get("relation"), list(RELATIONS)))
    if edge.get("confidence") not in CONFIDENCE:
        problems.append("%s: confidence %r is not one of %s" % (where, edge.get("confidence"), list(CONFIDENCE)))
    problem = _str_problem(edge.get("rationale"), "rationale", 1, 2000)
    if problem:
        problems.append("%s: %s" % (where, problem))
    if "observed_at" in edge:
        problem = _str_problem(edge["observed_at"], "observed_at", 1, 64)
        if problem:
            problems.append("%s: %s" % (where, problem))
    if edge.get("from") == edge.get("to") and isinstance(edge.get("from"), str):
        problems.append("%s: self-edge on %s" % (where, edge["from"]))
    return problems


def validate_document(doc, where):
    """Validate one shard/graph document in isolation (no cross-shard checks)."""
    problems = []
    if not isinstance(doc, dict):
        return ["%s: document must be a JSON object" % where]
    unknown = set(doc) - set(DOC_REQUIRED)
    if unknown:
        problems.append("%s: document has unknown top-level fields %s" % (where, sorted(unknown)))
    for field in DOC_REQUIRED:
        if field not in doc:
            problems.append("%s: document is missing %r" % (where, field))
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            "%s: schema_version must be %d, got %r" % (where, SCHEMA_VERSION, doc.get("schema_version"))
        )
    problem = _str_problem(doc.get("graph_id"), "graph_id", 1, 200)
    if problem:
        problems.append("%s: %s" % (where, problem))
    problem = _str_problem(doc.get("generated_at"), "generated_at", 1, 64)
    if problem:
        problems.append("%s: %s" % (where, problem))
    elif not ISO_DATE_RE.match(doc["generated_at"]):
        problems.append("%s: generated_at %r is not an ISO date" % (where, doc["generated_at"]))

    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        problems.append("%s: nodes must be an array" % where)
        nodes = []
    edges = doc.get("edges")
    if not isinstance(edges, list):
        problems.append("%s: edges must be an array" % where)
        edges = []

    for index, node in enumerate(nodes):
        problems.extend(validate_node(node, "%s nodes[%d]" % (where, index)))
    for index, edge in enumerate(edges):
        problems.extend(validate_edge(edge, "%s edges[%d]" % (where, index)))

    problems.extend("%s %s" % (where, item) for item in scan_secrets(doc))
    problems.extend("%s %s" % (where, item) for item in scan_private_paths(doc))
    return problems


def validate_corpus(documents, context=()):
    """Cross-document structural validation.

    `documents` is a sequence of (label, document) pairs. Edges may reference
    nodes declared in any document in the corpus.

    `context` is an optional sequence of (label, document) pairs whose nodes
    resolve edge endpoints but whose own nodes and edges are not re-validated.
    An evidence pack uses it to resolve references into the compiled graph.
    """
    problems = []
    owner = {}
    kinds = {}
    for label, doc in context:
        for node in doc.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                owner[node["id"]] = label
                kinds[node["id"]] = node.get("kind")
    context_ids = set(owner)
    declared_here = set()
    for label, doc in documents:
        for node in doc.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            if node_id in owner and node_id not in context_ids:
                problems.append(
                    "duplicate node id %s declared in %s and %s" % (node_id, owner[node_id], label)
                )
                continue
            owner[node_id] = label
            kinds[node_id] = node.get("kind")
            declared_here.add(node_id)

    seen_edges = set()
    result_relations = {}
    for label, doc in documents:
        for index, edge in enumerate(doc.get("edges", [])):
            if not isinstance(edge, dict):
                continue
            source = edge.get("from")
            target = edge.get("to")
            relation = edge.get("relation")
            where = "%s edges[%d] (%s -%s-> %s)" % (label, index, source, relation, target)
            if source not in owner:
                problems.append("%s: dangling edge; %r is not declared in any shard" % (where, source))
            if target not in owner:
                problems.append("%s: dangling edge; %r is not declared in any shard" % (where, target))
            key = (source, relation, target)
            if key in seen_edges:
                problems.append("%s: duplicate edge" % where)
            seen_edges.add(key)
            rule = ENDPOINT_RULES.get(relation)
            if rule and source in owner and target in owner:
                allowed_from, allowed_to = rule
                if kinds.get(source) not in allowed_from:
                    problems.append(
                        "%s: relation %r requires a %s source but %s is a %s"
                        % (where, relation, "|".join(sorted(allowed_from)), source, kinds.get(source))
                    )
                if kinds.get(target) not in allowed_to:
                    problems.append(
                        "%s: relation %r requires a %s target but %s is a %s"
                        % (where, relation, "|".join(sorted(allowed_to)), target, kinds.get(target))
                    )
            if kinds.get(source) == "applied_result" and relation in (
                RESULT_EXACTLY_ONE + RESULT_AT_LEAST_ONE
            ):
                result_relations.setdefault(source, {}).setdefault(relation, []).append(target)

    for node_id, kind in sorted(kinds.items()):
        if kind != "applied_result" or (node_id in context_ids and node_id not in declared_here):
            continue
        found = result_relations.get(node_id, {})
        for relation in RESULT_EXACTLY_ONE:
            targets = found.get(relation, [])
            if len(targets) != 1:
                problems.append(
                    "applied_result %s must have exactly one %r edge (found %d); the reified "
                    "hyperedge needs one mechanism, model, hardware, and workload anchor"
                    % (node_id, relation, len(targets))
                )
        for relation in RESULT_AT_LEAST_ONE:
            if not found.get(relation):
                problems.append(
                    "applied_result %s must have at least one %r edge naming the mechanism it applied"
                    % (node_id, relation)
                )
    return problems


def load_shards(shards_dir=None):
    """Load every shard under `shards_dir`, sorted by repo-relative path."""
    root = Path(shards_dir or SHARDS_DIR)
    documents = []
    for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
        label = path.relative_to(root).as_posix()
        documents.append((label, load_strict(path)))
    return documents


def merge(documents, graph_id, generated_at=None):
    """Merge shards into one deterministic graph document."""
    nodes = {}
    edges = {}
    latest = "0000-00-00"
    for _label, doc in documents:
        latest = max(latest, str(doc.get("generated_at", "")))
        for node in doc.get("nodes", []):
            nodes[node["id"]] = node
        for edge in doc.get("edges", []):
            edges[(edge["from"], edge["relation"], edge["to"])] = edge
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "generated_at": generated_at or latest,
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
    }
