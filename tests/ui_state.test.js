const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const uiState = require(path.join(__dirname, '..', 'conclave', 'static', 'ui_state.js'));

test('buildPhaseState marks done + active phases', () => {
  const run = {
    status: 'running',
    events: [
      { phase: 'route', status: 'done' },
      { phase: 'retrieve', status: 'done' },
      { phase: 'model', role: 'reasoner' },
    ],
  };
  const phases = uiState.buildPhaseState(run);
  const byKey = Object.fromEntries(phases.map((p) => [p.key, p]));
  assert.equal(byKey.route.done, true);
  assert.equal(byKey.retrieve.done, true);
  assert.equal(byKey.reasoner.done, true);
  assert.equal(byKey.critic.active, true);
});

test('summarizeEvent formats deliberation result', () => {
  const run = { events: [] };
  const event = { phase: 'deliberate', status: 'round_result', round: 1, agreement: true, weighted_ratio: 0.72 };
  const summary = uiState.summarizeEvent(event, run, 0);
  assert.match(summary, /Round 1/);
  assert.match(summary, /white smoke/);
  assert.match(summary, /ratio 72%/);
});
