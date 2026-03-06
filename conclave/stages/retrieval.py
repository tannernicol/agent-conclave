from __future__ import annotations

from typing import Any, Dict, List, Optional


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def retrieve_context(pipeline, query: str, route: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rag_results: List[Dict[str, Any]] = []
    rag_cfg = pipeline.config.rag
    max_per_collection = int(rag_cfg.get("max_results_per_collection", 8))
    prefer_non_pdf = _coerce_bool(rag_cfg.get("prefer_non_pdf", False))
    semantic = rag_cfg.get("semantic")
    disable_domains = set(rag_cfg.get("disable_domains", []) or [])
    if route.get("domain") not in disable_domains:
        for coll in route.get("collections", []):
            rag_results.extend(pipeline.rag.search(query, collection=coll, limit=max_per_collection, semantic=semantic))
    rag_results = pipeline._filter_rag_results(rag_results)
    allowlist = route.get("allowlist") or []
    if allowlist:
        rag_results = [item for item in rag_results if item.get("collection") in allowlist]
    index_results = []
    file_results = pipeline.rag.search_files(query, limit=10)
    if pipeline.config.index.get("enabled", True):
        auto_build = bool(pipeline.config.index.get("auto_build", False))
        if pipeline.index.db_path.exists() or auto_build:
            pipeline._maybe_refresh_index()
            index_results = pipeline.index.search(query, limit=10)
    combined_files = file_results + index_results
    if prefer_non_pdf:
        # When prefer_non_pdf=True we want non-PDF sources first, so PDFs sort last.
        rag_results.sort(key=lambda x: str(x.get("path") or x.get("name") or "").lower().endswith(".pdf"))
    user_inputs = pipeline._load_user_input(meta)
    if not user_inputs and query:
        user_inputs = [{
            "title": "prompt",
            "snippet": query,
            "full_text": query,
            "collection": "user-input",
            "source": "user",
        }]
    source_items: List[Dict[str, Any]] = []
    source_errors: List[Dict[str, Any]] = []
    domain = route.get("domain")
    input_artifacts: List[Dict[str, Any]] = []
    instructions = user_inputs[0].get("full_text") if user_inputs else ""
    artifact_paths = pipeline._extract_artifact_paths(str(instructions))
    if artifact_paths:
        input_artifacts = pipeline._summarize_artifacts(artifact_paths)
        if input_artifacts:
            source_items.extend(input_artifacts)

    output_type = meta.get("output_type") if meta else None
    image_paths = [item.get("path") for item in input_artifacts if item.get("kind") == "image" and item.get("path")]
    vision_summary_text = ""
    if image_paths and pipeline._output_requires(output_type, "image_understanding"):
        vision_prompt = (
            "Summarize the attached photos for design decisions. "
            "Note lighting, dominant materials, current cabinet color/finish, wall/trim colors, "
            "flooring, counters, and any constraints that affect cabinet color choices."
        )
        vision_summary, provider = pipeline._vision_summary(vision_prompt, image_paths)
        if vision_summary:
            vision_summary_text = vision_summary
            if provider:
                pipeline._record_vision_usage(provider, len(image_paths))
            source_items.append({
                "path": f"{provider}://vision/summary",
                "title": f"Vision Summary ({provider})",
                "snippet": vision_summary[:1600],
                "collection": f"vision-{provider}",
                "source": "vision",
            })

    previous = pipeline._latest_for_meta(meta)
    previous_run = None
    if previous and previous.get("consensus", {}).get("answer"):
        previous_run = {
            "id": previous.get("id"),
            "created_at": previous.get("created_at"),
            "agreement": (previous.get("artifacts") or {}).get("deliberation", {}).get("agreement"),
            "answer": str(previous.get("consensus", {}).get("answer", ""))[:2400],
        }

    on_demand = pipeline.verifier.fetch(domain or "general", query)
    if on_demand.items:
        source_items.extend(on_demand.items)
    if on_demand.errors:
        source_errors.extend(on_demand.errors)
    if pipeline._audit and (on_demand.items or on_demand.errors):
        pipeline._audit.log("sources.on_demand", {
            "domain": domain,
            "items": len(on_demand.items),
            "errors": on_demand.errors,
        })

    evidence_limit = None
    if meta and meta.get("evidence_limit"):
        try:
            evidence_limit = int(meta.get("evidence_limit"))
        except Exception:
            evidence_limit = None
    required_collections = route.get("required_collections", [])
    evidence, stats = pipeline._select_evidence(
        rag_results,
        combined_files,
        limit=evidence_limit or 12,
        preferred_collections=route.get("collections", []),
        required_collections=required_collections,
        domain=route.get("domain"),
        domain_paths=pipeline.config.quality.get("domain_paths", {}),
        collection_reliability=pipeline.config.rag.get("collection_reliability", {}),
        user_items=user_inputs,
        source_items=source_items,
    )
    rag_errors = pipeline.rag.drain_errors()
    if rag_errors:
        stats["rag_errors"] = rag_errors
    if source_errors:
        stats["source_errors"] = source_errors
    if on_demand.items:
        stats["on_demand_count"] = len(on_demand.items)
    if on_demand.errors:
        stats["on_demand_errors"] = on_demand.errors
    if user_inputs:
        stats["input_path"] = user_inputs[0].get("path")
    result = {
        "rag": rag_results,
        "file_index": combined_files,
        "sources": source_items,
        "evidence": evidence,
        "stats": stats,
        "user_inputs": user_inputs,
        "input_artifacts": input_artifacts,
        "previous_run": previous_run,
        "agent_sync": pipeline._agent_sync_summary(),
    }
    if vision_summary_text:
        result["vision_summary"] = vision_summary_text
    return result
