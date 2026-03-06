/* WebSocket helpers with retry + fallback. */

(function initConclaveWS() {
  function openRunSocket(options) {
    const {
      runId,
      onState,
      onEvent,
      onFallback,
      onStatus,
      maxRetries = 3,
      baseDelayMs = 800,
      maxDelayMs = 6000,
    } = options || {};

    if (!runId) return { close: () => {} };

    let socket = null;
    let retries = 0;
    let closedByClient = false;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/run/${encodeURIComponent(runId)}`;

    const backoff = () => Math.min(maxDelayMs, baseDelayMs * Math.pow(2, retries));

    const connect = () => {
      if (closedByClient) return;
      onStatus && onStatus({ state: 'connecting', retries });
      socket = new WebSocket(wsUrl);

      socket.onmessage = (event) => {
        let payload = null;
        try {
          payload = JSON.parse(event.data);
        } catch (err) {
          onStatus && onStatus({ state: 'error', error: 'invalid_json' });
          return;
        }
        if (payload.type === 'state' || payload.type === 'complete') {
          onState && onState(payload.run);
        } else if (payload.type === 'event') {
          onEvent && onEvent(payload.event);
        }
      };

      socket.onerror = () => {
        onStatus && onStatus({ state: 'error', error: 'socket_error' });
      };

      socket.onclose = () => {
        if (closedByClient) return;
        if (retries < maxRetries) {
          retries += 1;
          onStatus && onStatus({ state: 'retrying', retries });
          setTimeout(connect, backoff());
          return;
        }
        onStatus && onStatus({ state: 'fallback', retries });
        onFallback && onFallback();
      };
    };

    connect();

    return {
      close: () => {
        closedByClient = true;
        if (socket) socket.close();
      },
    };
  }

  window.ConclaveWS = { openRunSocket };
})();
