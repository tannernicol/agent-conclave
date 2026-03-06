/* Pure UI state helpers for Conclave. */

function buildPhaseState(run) {
  const phases = [
    { key: 'route', label: 'Route' },
    { key: 'retrieve', label: 'RAG' },
    { key: 'reasoner', label: 'Reason' },
    { key: 'critic', label: 'Critic' },
    { key: 'summarizer', label: 'Summary' },
  ];
  if (!run) return phases;

  const events = run.events || [];
  const done = new Set();

  events.forEach((event) => {
    if (event.phase === 'route' && event.status === 'done') done.add('route');
    if (event.phase === 'retrieve' && event.status === 'done') done.add('retrieve');
    if (event.phase === 'model' && event.role) {
      const roleKey = event.role === 'critic_panel' ? 'critic' : event.role;
      done.add(roleKey);
    }
  });

  if (run.status === 'complete') {
    phases.forEach((phase) => done.add(phase.key));
  }

  let active = null;
  if (run.status === 'running') {
    active = phases.find((phase) => !done.has(phase.key))?.key || null;
  }

  return phases.map((phase) => ({
    ...phase,
    done: done.has(phase.key),
    active: active === phase.key,
  }));
}

function summarizeEvent(event, run, index) {
  if (!event) return '';
  const phase = event.phase || 'run';
  if (phase === 'model') {
    const role = event.role || 'model';
    const label = event.model_label || event.model_id || 'model';
    const duration = event.duration_ms ? `${Math.round(event.duration_ms / 100) / 10}s` : '';
    const status = event.ok === false ? 'error' : (event.ok === true ? 'ok' : '');
    const error = event.error || (event.ok === false ? 'failed' : '');

    let description = `${role}: ${label}`;
    if (role === 'reasoner') {
      const prior = (run?.events || []).slice(0, index).reverse().find((item) => item.phase === 'deliberate' && item.disagreements);
      const count = prior?.disagreements?.length || 0;
      description = count > 0
        ? `Reasoner revising after ${count} critique${count === 1 ? '' : 's'}`
        : 'Reasoner drafting initial response';
    } else if (role === 'critic_panel') {
      description = `Panel critique from ${label}`;
    } else if (role === 'critic') {
      description = `Primary critic review (${label})`;
    } else if (role === 'summarizer') {
      description = 'Summarizer distilling consensus';
    }

    const statusLabel = status ? ` • ${status}` : '';
    const errorLabel = status === 'error' && error ? ` • ${error}` : '';
    return `${description}${duration ? ` • ${duration}` : ''}${statusLabel}${errorLabel}`.trim();
  }
  if (phase === 'retrieve') {
    const evidence = event.context?.evidence;
    const ragCount = event.context?.rag;
    const fileCount = event.context?.file_index;
    const details = [];
    if (typeof evidence === 'number') details.push(`evidence ${evidence}`);
    if (typeof ragCount === 'number') details.push(`rag ${ragCount}`);
    if (typeof fileCount === 'number') details.push(`files ${fileCount}`);
    return `Retrieval completed${details.length ? ` • ${details.join(', ')}` : ''}`.trim();
  }
  if (phase === 'route') {
    if (event.status === 'done') {
      const plan = event.models || (event.route && event.route.plan_details) || {};
      const roles = [];
      const pick = (key, label) => {
        const info = plan?.[key];
        if (info && (info.label || info.id || info)) {
          roles.push(`${label}:${info.label || info.id || info}`);
        }
      };
      pick('reasoner', 'reasoner');
      pick('critic', 'critic');
      pick('summarizer', 'summarizer');
      return `Routing complete${roles.length ? ` • ${roles.join(', ')}` : ''}`;
    }
    return 'Routing models';
  }
  if (phase === 'quality') {
    const count = event.evidence_count;
    const signal = event.max_signal_score;
    const issues = event.issues;
    const details = [];
    if (typeof count === 'number') details.push(`evidence ${count}`);
    if (typeof signal === 'number') details.push(`signal ${signal.toFixed(2)}`);
    if (Array.isArray(issues) && issues.length) details.push(`${issues.length} issue${issues.length === 1 ? '' : 's'}`);
    return `Quality check${details.length ? ` • ${details.join(', ')}` : ''}`.trim();
  }
  if (phase === 'deliberate') {
    const roundLabel = event.round ? `Round ${event.round}` : 'Deliberation';
    const label = event.model_label || event.model_id || '';
    const duration = typeof event.duration_s === 'number' ? `${event.duration_s.toFixed(1)}s` : '';
    const summary = (event.summary || '').trim();
    const disagreementCount = (event.disagreements || []).length;
    if (event.status === 'round_start') {
      return `${roundLabel}: started`;
    }
    if (event.status === 'reasoner_start') {
      return `${roundLabel}: reasoner thinking${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'reasoner_done') {
      return `${roundLabel}: reasoner draft${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'critic_start') {
      return `${roundLabel}: critic reviewing${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'critic_done') {
      const verdict = event.verdict ? event.verdict.toUpperCase() : '';
      return `${roundLabel}: critic ${verdict || 'done'}${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'panel_start') {
      return `${roundLabel}: panel reviewing${label ? ` (${label})` : ''}`;
    }
    if (event.status === 'panel_done') {
      const verdict = event.verdict ? event.verdict.toUpperCase() : '';
      return `${roundLabel}: panel ${verdict || 'done'}${duration ? ` • ${duration}` : ''}${summary ? ` • ${summary}` : ''}`;
    }
    if (event.status === 'round_result') {
      const smoke = event.agreement ? '\u2601\ufe0f white smoke' : '\u2587\u2587 black smoke';
      const verdict = event.agreement ? 'AGREE' : 'DISAGREE';
      const issues = disagreementCount ? ` • ${disagreementCount} issue${disagreementCount === 1 ? '' : 's'}` : '';
      const ratio = typeof event.weighted_ratio === 'number' ? ` • ratio ${(event.weighted_ratio * 100).toFixed(0)}%` : '';
      return `${roundLabel}: ${smoke} ${verdict}${issues}${ratio}`;
    }
    if (event.status === 'stable') {
      return `${roundLabel}: \u2587\u2587 black smoke — stable disagreement`;
    }
    if (event.status === 'stop') {
      return `${roundLabel}: stopping (${event.reason || 'done'})`;
    }
    if (event.agreement === true) {
      return `${roundLabel}: consensus reached`;
    }
    if (disagreementCount > 0) {
      return `${roundLabel}: critic pushed back on ${disagreementCount} point${disagreementCount === 1 ? '' : 's'}`;
    }
    if (event.status === 'start') {
      return `${roundLabel}: panel reviewing`;
    }
    return `${roundLabel}: continuing negotiation`;
  }
  if (phase === 'requirements') {
    return event.status === 'ok' ? 'Requirements satisfied' : 'Checking requirements';
  }
  if (phase === 'preflight') {
    return 'Preflight checks';
  }
  return `${phase} ${event.status || ''}`.trim();
}

if (typeof window !== 'undefined') {
  window.ConclaveUI = { buildPhaseState, summarizeEvent };
}

if (typeof module !== 'undefined') {
  module.exports = { buildPhaseState, summarizeEvent };
}
