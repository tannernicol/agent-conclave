from __future__ import annotations

from typing import Any, Dict, List
import time

from conclave.domains import get_domain_instructions


def _bounded_timeout_seconds(model_timeout: Any, budget: float | None) -> int | None:
    if budget is not None:
        return max(1, int(min(float(model_timeout or budget), max(0.0, budget))))
    if model_timeout:
        try:
            return max(1, int(model_timeout))
        except Exception:
            return None
    return None


def _effective_min_time_left_seconds(configured_seconds: Any, run_timeout_seconds: Any) -> float:
    try:
        configured = float(configured_seconds or 0)
    except Exception:
        configured = 0.0
    try:
        run_timeout = float(run_timeout_seconds or 0)
    except Exception:
        run_timeout = 0.0
    if configured <= 0 or run_timeout <= 0:
        return max(0.0, configured)
    short_run_cap = max(15.0, run_timeout * 0.25)
    return max(0.0, min(configured, short_run_cap))


def _merge_disagreements(*groups: List[str], limit: int | None = None) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            text = str(item).strip()
            norm = " ".join(text.lower().split())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(text)
            if limit is not None and len(merged) >= limit:
                return merged
    return merged


def deliberate(pipeline, query: str, context: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    plan = route.get("plan", {})
    reasoner_model = plan.get("creator") or plan.get("reasoner") or next(iter(plan.values()), None)
    critic_model = plan.get("critic") or plan.get("reviewer") or reasoner_model
    config = pipeline.config.raw.get("deliberation", {})
    max_rounds = int(config.get("max_rounds", 3))
    require_agreement = bool(config.get("require_agreement", True))
    stop_on_repeat = bool(config.get("stop_on_repeat_disagreements", False))
    stability_rounds = int(config.get("stability_rounds", 0))
    max_draft_chars = int(config.get("max_draft_chars", 4000))
    max_feedback_chars = int(config.get("max_feedback_chars", 4000))
    max_disagreements = int(config.get("max_disagreements_per_review", 8))
    model_timeout = config.get("model_timeout_seconds")
    panel_cfg = config.get("panel", {}) if isinstance(config, dict) else {}
    panel_enabled = bool(panel_cfg.get("enabled", False))
    panel_review_on_agreement = bool(panel_cfg.get("review_on_agreement", False))
    panel_models = list(route.get("panel_models") or [])
    panel_require_all = bool(panel_cfg.get("require_all", True))
    panel_min_ratio = panel_cfg.get("min_agree_ratio") if isinstance(panel_cfg, dict) else None
    panel_timeout = panel_cfg.get("timeout_seconds") if isinstance(panel_cfg, dict) else None
    panel_max_rounds = panel_cfg.get("max_rounds") if isinstance(panel_cfg, dict) else None
    try:
        panel_start_round = max(1, int(panel_cfg.get("escalate_after_round", 1)))
    except Exception:
        panel_start_round = 1
    priority_models = pipeline._priority_models(config)
    priority_require_all = bool(config.get("priority_require_all", True))
    priority_min_ratio = config.get("priority_min_ratio")
    if route.get("panel_require_all") is not None:
        panel_require_all = bool(route.get("panel_require_all"))
    if route.get("panel_min_ratio") is not None:
        try:
            panel_min_ratio = float(route.get("panel_min_ratio"))
        except Exception:
            pass
    if getattr(pipeline, "_is_high_importance", None) and pipeline._is_high_importance():
        panel_enabled = panel_enabled or bool(panel_models)
        panel_review_on_agreement = True
    min_time_left = _effective_min_time_left_seconds(
        config.get("min_time_left_seconds", 0),
        pipeline.config.run_timeout_seconds,
    )
    context_blob = pipeline._format_context(context)
    runtime_blob = pipeline._format_runtime_context(route, context)
    instructions = pipeline._user_instructions(context)
    previous_run = context.get("previous_run") or {}
    prev_answer = previous_run.get("answer") or ""
    domain = route.get("domain")
    domain_hints = get_domain_instructions(domain)
    domain_instructions = domain_hints.deliberation_hint
    output_instructions = pipeline._output_instructions(context.get("output_type"))
    output_meta = context.get("output") or {}
    missing_caps = output_meta.get("missing") or []
    tool_guard = (
        "Do not use tools, run shell commands, or attempt network access. "
        "Respond directly with your analysis."
    )
    rounds = []
    panel_rounds: List[Dict[str, Any]] = []
    diversity_checks: List[Dict[str, Any]] = []
    diversity_calls = 0
    reasoner_out = ""
    critic_out = ""
    review_out = ""
    agreement = False
    required_set = set(pipeline._required_model_ids())
    stable_signature = ""
    stable_count = 0
    stop_reason: str | None = None
    previous_disagreements: List[str] = []
    open_disagreements: List[str] = []
    all_disagreements: List[str] = []

    def _emit_deliberate(payload: Dict[str, Any]) -> None:
        if pipeline._run_id:
            pipeline.store.append_event(pipeline._run_id, {"phase": "deliberate", **payload})

    _summary_line = pipeline._summary_line

    for round_idx in range(1, max_rounds + 1):
        _emit_deliberate({"status": "round_start", "round": round_idx, "max_rounds": max_rounds})
        remaining = pipeline._time_remaining()
        if min_time_left and (remaining or 0) < min_time_left:
            stop_reason = "timeout_guard"
            break
        budget = None
        if remaining is not None:
            budget = max(0.0, remaining - float(min_time_left))
        per_call_timeout = _bounded_timeout_seconds(model_timeout, budget)
        analysis_prompt = (
            "You are the reasoner. Provide a careful analysis and propose a decision.\n"
            "Be specific and prescriptive. If evidence is weak, proceed with assumptions and mark confidence low.\n"
            "Do not refuse or defer just because data is missing.\n"
            "Negotiate toward consensus: address each disagreement explicitly and show what changed.\n"
            "Separate product recommendation from Conclave mechanics confirmation; confirmation failures are constraints, not blockers.\n\n"
            f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n"
        )
        if round_idx > 1:
            draft_excerpt = pipeline._truncate_text(reasoner_out, max_draft_chars)
            critic_excerpt = pipeline._truncate_text(review_out or critic_out, max_feedback_chars)
            unresolved = previous_disagreements[:max_disagreements]
            unresolved_blob = ""
            if unresolved:
                unresolved_blob = "\nUnresolved disagreements (must address each):\n"
                for item in unresolved:
                    unresolved_blob += f"- {item}\n"
            analysis_prompt = (
                "You are the reasoner. Revise the decision to address the critic's feedback.\n\n"
                f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n\n"
                f"{unresolved_blob}\n"
                f"Previous draft:\n{draft_excerpt}\n\n"
                f"Critic feedback:\n{critic_excerpt}\n"
            )
            analysis_prompt += (
                "\nInclude a 'Resolution log' section that lists each prior disagreement and marks each as RESOLVED or ACCEPTED (with tradeoff).\n"
                "If a disagreement is about runtime confirmation, explicitly mark it as ACCEPTED/FAILED with a remediation step.\n"
            )
        if domain_instructions or instructions or output_instructions or tool_guard:
            analysis_prompt += f"\n{domain_instructions}\n"
            analysis_prompt += f"\n{tool_guard}\n"
            if output_instructions:
                analysis_prompt += f"\n{output_instructions}\n"
            if missing_caps:
                analysis_prompt += (
                    "\nCapability gaps detected: "
                    f"{', '.join(missing_caps)}. Provide a best-effort answer, "
                    "note the limitation, and include a concrete plan for generating the missing artifacts if tools are later enabled.\n"
                )
            if instructions:
                analysis_prompt += f"Instructions from input:\n{instructions}\n"
        if prev_answer:
            analysis_prompt += (
                "\nPrevious consensus (Version N-1):\n"
                f"{pipeline._truncate_text(prev_answer, 1200)}\n"
                "Reconcile with the prior result. Include a short 'Reconciliation' section noting what changed and why. "
                "If no material change, say 'No change' and explain stability.\n"
            )
        _emit_deliberate({
            "status": "reasoner_start",
            "round": round_idx,
            "max_rounds": max_rounds,
            "role": "reasoner",
            "model_id": reasoner_model,
            "model_label": pipeline._model_label(reasoner_model) if reasoner_model else None,
            "timeout_s": per_call_timeout,
        })
        reasoner_start = time.perf_counter()
        reasoner_out = pipeline._call_model(reasoner_model, analysis_prompt, role="reasoner", timeout_seconds=per_call_timeout)
        reasoner_duration = time.perf_counter() - reasoner_start
        _emit_deliberate({
            "status": "reasoner_done",
            "round": round_idx,
            "max_rounds": max_rounds,
            "role": "reasoner",
            "model_id": reasoner_model,
            "model_label": pipeline._model_label(reasoner_model) if reasoner_model else None,
            "duration_s": round(reasoner_duration, 2),
            "summary": _summary_line(reasoner_out),
        })

        critic_prompt = (
            "You are the critic. Challenge the reasoning, list disagreements and gaps, and suggest fixes.\n"
            "Do not reject the task as out-of-scope; focus on improving the draft.\n"
            "Do not introduce brand new disagreements in later rounds unless they are critical.\n"
            "Negotiate toward consensus: if the draft clearly acknowledges an inherent limitation and proposes a concrete verification step, treat that item as resolved.\n"
            "Evaluate product recommendation separately from Conclave mechanics confirmation; if confirmation items are marked Failed/Unverified with remediation, do not treat as blocking.\n"
            "If remaining issues are minor or non-blocking, respond with AGREE and list up to 3 minor follow-ups.\n"
            "List at most 3 disagreements; consolidate overlapping items.\n"
            "Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\nAGREE or DISAGREE\n\n"
            f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nReasoner draft:\n{reasoner_out}\n"
        )
        if round_idx > 1 and previous_disagreements:
            critic_prompt += "\nPrevious disagreements (resolve if addressed; only list unresolved):\n"
            for item in previous_disagreements[:max_disagreements]:
                critic_prompt += f"- {item}\n"
            critic_prompt += "\nIf all prior disagreements are resolved, respond with AGREE.\n"
        if domain_instructions or instructions or output_instructions or tool_guard:
            critic_prompt += f"\n{domain_instructions}\n"
            critic_prompt += f"\n{tool_guard}\n"
            if output_instructions:
                critic_prompt += f"\n{output_instructions}\n"
            if missing_caps:
                critic_prompt += (
                    "\nCapability gaps detected: "
                    f"{', '.join(missing_caps)}. Ensure the draft acknowledges the limitation and provides a fallback.\n"
                )
            if instructions:
                critic_prompt += f"Instructions from input:\n{instructions}\n"
        if prev_answer:
            critic_prompt += (
                "\nPrevious consensus (Version N-1):\n"
                f"{pipeline._truncate_text(prev_answer, 800)}\n"
                "Ensure the draft reconciles with prior results and clearly justifies any changes.\n"
            )
        _emit_deliberate({
            "status": "critic_start",
            "round": round_idx,
            "max_rounds": max_rounds,
            "role": "critic",
            "model_id": critic_model,
            "model_label": pipeline._model_label(critic_model) if critic_model else None,
            "timeout_s": per_call_timeout,
        })
        critic_start = time.perf_counter()
        critic_out = pipeline._call_model(critic_model, critic_prompt, role="critic", timeout_seconds=per_call_timeout)
        critic_duration = time.perf_counter() - critic_start
        critic_agreement = pipeline._critic_agrees(critic_out)
        critic_disagreements = pipeline._extract_disagreements(critic_out)
        _emit_deliberate({
            "status": "critic_done",
            "round": round_idx,
            "max_rounds": max_rounds,
            "role": "critic",
            "model_id": critic_model,
            "model_label": pipeline._model_label(critic_model) if critic_model else None,
            "duration_s": round(critic_duration, 2),
            "verdict": "agree" if critic_agreement else "disagree",
            "summary": _summary_line(critic_out),
        })

        agreement = critic_agreement
        panel_agreement = None
        weighted_ratio = None
        round_disagreements = list(critic_disagreements)
        review_out = critic_out

        panel_window_open = panel_enabled and bool(panel_models) and round_idx >= panel_start_round
        if panel_window_open and panel_max_rounds is not None:
            try:
                panel_window_open = round_idx < (panel_start_round + int(panel_max_rounds))
            except Exception:
                pass
        should_run_panel = bool(panel_window_open and (not critic_agreement or panel_review_on_agreement))

        if should_run_panel:
            panel_prompt = (
                "You are a panel reviewer. Read the reasoner's draft and the critic's review.\n"
                "Decide whether this answer is ready to ship without human arbitration.\n"
                "Only raise new disagreements if they are material and would change the outcome.\n"
                "If the critic's objections are resolved or non-blocking, respond with AGREE.\n"
                "List at most 3 disagreements and at most 3 follow-ups.\n"
                "Return sections:\nDisagreements:\n- ...\nFollow-ups:\n- ...\nVerdict:\nAGREE or DISAGREE\n\n"
                f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nReasoner draft:\n{reasoner_out}\n\nCritic review:\n{critic_out}\n"
            )
            if round_idx > 1 and previous_disagreements:
                panel_prompt += "\nPrevious disagreements (only keep unresolved ones):\n"
                for item in previous_disagreements[:max_disagreements]:
                    panel_prompt += f"- {item}\n"
            if domain_instructions or instructions or output_instructions or tool_guard:
                panel_prompt += f"\n{domain_instructions}\n"
                panel_prompt += f"\n{tool_guard}\n"
                if output_instructions:
                    panel_prompt += f"\n{output_instructions}\n"
                if instructions:
                    panel_prompt += f"Instructions from input:\n{instructions}\n"
            if prev_answer:
                panel_prompt += (
                    "\nPrevious consensus (Version N-1):\n"
                    f"{pipeline._truncate_text(prev_answer, 800)}\n"
                    "Prefer stability unless the current draft has a materially better justification.\n"
                )

            panel_reviews = []
            next_panel_models = []
            for model_id in panel_models:
                effective_panel_timeout = panel_timeout
                if per_call_timeout is not None:
                    if effective_panel_timeout is None:
                        effective_panel_timeout = per_call_timeout
                    else:
                        effective_panel_timeout = min(int(effective_panel_timeout), int(per_call_timeout))
                _emit_deliberate({
                    "status": "panel_start",
                    "round": round_idx,
                    "max_rounds": max_rounds,
                    "role": "critic_panel",
                    "model_id": model_id,
                    "model_label": pipeline._model_label(model_id),
                    "timeout_s": effective_panel_timeout,
                })
                panel_start = time.perf_counter()
                review = pipeline._call_model(model_id, panel_prompt, role="critic_panel", timeout_seconds=effective_panel_timeout)
                panel_duration = time.perf_counter() - panel_start
                meta = pipeline._last_model_results.get(model_id, {})
                ok = meta.get("ok", True)
                error = meta.get("error")
                stderr = meta.get("stderr")
                skip_optional = False
                if not ok and model_id == "cli:gemini":
                    stderr_lower = (stderr or "").lower()
                    quota_hint = any(token in stderr_lower for token in ("quota", "rate limit", "capacity", "exhausted", "429"))
                    if error == "timeout" or quota_hint:
                        skip_optional = True
                verdict = pipeline._critic_agrees(review) if ok else False
                verdict_label = "agree" if verdict else ("skipped" if skip_optional else ("error" if not ok else "disagree"))
                _emit_deliberate({
                    "status": "panel_done",
                    "round": round_idx,
                    "max_rounds": max_rounds,
                    "role": "critic_panel",
                    "model_id": model_id,
                    "model_label": pipeline._model_label(model_id),
                    "duration_s": round(panel_duration, 2),
                    "verdict": verdict_label,
                    "summary": _summary_line(review),
                })
                panel_reviews.append({
                    "model_id": model_id,
                    "label": pipeline._model_label(model_id),
                    "verdict": verdict_label,
                    "ok": ok,
                    "error": error,
                    "stderr": stderr,
                    "skipped": skip_optional,
                    "disagreements": pipeline._extract_disagreements(review) if ok else ([] if skip_optional else ([f"model failed: {error}"] if error else [])),
                    "text": review,
                })
                if ok or model_id in required_set:
                    next_panel_models.append(model_id)
            if next_panel_models != panel_models:
                panel_models = next_panel_models
            panel_agreement = pipeline._panel_agreement(
                panel_reviews,
                require_all=panel_require_all,
                min_ratio=None,
                priority_models=priority_models,
                priority_require_all=priority_require_all,
                priority_min_ratio=priority_min_ratio,
            )
            if panel_min_ratio is not None and not panel_require_all:
                try:
                    panel_agreement = pipeline._panel_agreement(
                        panel_reviews,
                        require_all=False,
                        min_ratio=float(panel_min_ratio),
                        priority_models=priority_models,
                        priority_require_all=priority_require_all,
                        priority_min_ratio=priority_min_ratio,
                    )
                except Exception:
                    panel_agreement = pipeline._panel_agreement(
                        panel_reviews,
                        require_all=panel_require_all,
                        min_ratio=None,
                        priority_models=priority_models,
                        priority_require_all=priority_require_all,
                        priority_min_ratio=priority_min_ratio,
                    )
            panel_feedback = pipeline._format_panel_feedback(panel_reviews, max_disagreements=max_disagreements)
            review_out = f"{critic_out}\n\n{panel_feedback}".strip()
            filtered_disagreements = pipeline._aggregate_panel_disagreements(
                panel_reviews,
                priority_models=priority_models,
            )
            round_disagreements = _merge_disagreements(
                critic_disagreements,
                filtered_disagreements,
                limit=max_disagreements,
            )
            weighted_agrees = sum(
                pipeline._model_confidence_weight(r.get("model_id", ""))
                for r in panel_reviews if r.get("verdict") == "agree" and r.get("ok", True)
            )
            total_weight = sum(
                pipeline._model_confidence_weight(r.get("model_id", ""))
                for r in panel_reviews if r.get("ok", True)
            )
            weighted_ratio = round(weighted_agrees / total_weight, 3) if total_weight > 0 else 0.0
            agreement = bool(critic_agreement and panel_agreement)
            panel_rounds.append({
                "round": round_idx,
                "agreement": agreement,
                "critic_agreement": critic_agreement,
                "panel_agreement": panel_agreement,
                "weighted_ratio": weighted_ratio,
                "reviews": panel_reviews,
            })

        round_entry = {
            "round": round_idx,
            "agreement": agreement,
            "critic_agreement": critic_agreement,
            "disagreements": round_disagreements,
        }
        if panel_agreement is not None:
            round_entry["panel_agreement"] = panel_agreement
        if weighted_ratio is not None:
            round_entry["weighted_ratio"] = weighted_ratio
        rounds.append(round_entry)
        diversity_entry = pipeline._maybe_run_diversity_check(
            query=query,
            context_blob=context_blob,
            reasoner_out=reasoner_out,
            critic_out=review_out or critic_out,
            round_idx=round_idx,
            max_rounds=max_rounds,
            agreement=agreement,
            domain_instructions=domain_instructions,
            output_instructions=output_instructions,
            instructions=instructions,
            calls_so_far=diversity_calls,
        )
        if diversity_entry:
            diversity_checks.append(diversity_entry)
            diversity_calls += 1
        _emit_deliberate({"status": "round_result", **round_entry})
        previous_disagreements = list(round_entry.get("disagreements") or [])
        round_disagreements = list(round_entry.get("disagreements") or [])
        if round_disagreements:
            open_disagreements = list(round_disagreements)
            all_disagreements = _merge_disagreements(all_disagreements, round_disagreements)
        else:
            open_disagreements = []
        signature = pipeline._disagreement_signature(round_entry.get("disagreements") or [])
        if stop_on_repeat and stability_rounds > 0 and signature:
            if signature == stable_signature:
                stable_count += 1
            else:
                stable_signature = signature
                stable_count = 1
            if stable_count >= stability_rounds and not agreement:
                round_entry["stopped_reason"] = "stable_disagreements"
                stop_reason = "stable_disagreements"
                _emit_deliberate({
                    "status": "stable",
                    "round": round_idx,
                    "max_rounds": max_rounds,
                    "consecutive": stable_count,
                    "reason": stop_reason,
                })
                break
        if agreement or not require_agreement:
            break

    if stop_reason:
        _emit_deliberate({
            "status": "stop",
            "round": round_idx,
            "max_rounds": max_rounds,
            "reason": stop_reason,
        })

    result = {
        "reasoner": reasoner_out,
        "critic": critic_out,
        "disagreements": open_disagreements,
        "all_disagreements": all_disagreements,
        "rounds": rounds,
        "agreement": agreement,
        "stopped_reason": stop_reason,
        "panel": panel_rounds,
        "panel_models": panel_models,
        "diversity": diversity_checks,
    }
    # Add quality score to deliberation result
    result["quality_score"] = pipeline._deliberation_score(result)
    return result
