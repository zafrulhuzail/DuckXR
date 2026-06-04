const http = require('http');

const PORT = Number(process.env.PORT || 8080);
const HOST = process.env.HOST || '0.0.0.0';

function sendJson(res, statusCode, body) {
  const json = JSON.stringify(body, null, 2);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'POST, OPTIONS, GET'
  });
  res.end(json);
}

function safeString(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function buildReply(payload) {
  const message = safeString(payload.message);
  const duckName = safeString(payload.duckName) || 'Ducky';
  const sceneName = safeString(payload.sceneName);

  if (!message) {
    return `${duckName} here. I didn't get a message yet.`;
  }

  return `${duckName} heard: "${message}"${sceneName ? ` in scene ${sceneName}` : ''}. Backend is working.`;
}

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') {
    return sendJson(res, 204, {});
  }

  if (req.method === 'GET' && req.url === '/health') {
    return sendJson(res, 200, {
      ok: true,
      service: 'duckxr-backend',
      port: PORT
    });
  }

  if (req.method === 'POST' && req.url === '/duck/chat') {
    let raw = '';

    req.on('data', chunk => {
      raw += chunk;
      if (raw.length > 1024 * 1024) {
        req.destroy();
      }
    });

    req.on('end', () => {
      let payload;

      try {
        payload = raw ? JSON.parse(raw) : {};
      } catch (error) {
        return sendJson(res, 400, {
          success: false,
          sessionId: '',
          reply: '',
          shouldSpeak: false,
          mood: 'error',
          hints: [],
          error: 'Invalid JSON body'
        });
      }

      const message = safeString(payload.message);
      if (!message) {
        return sendJson(res, 400, {
          success: false,
          sessionId: safeString(payload.sessionId),
          reply: '',
          shouldSpeak: false,
          mood: 'error',
          hints: [],
          error: 'message is required'
        });
      }

      const sessionId = safeString(payload.sessionId) || `duckxr-${Date.now()}`;
      const reply = buildReply({ ...payload, message });

      return sendJson(res, 200, {
        success: true,
        sessionId,
        reply,
        shouldSpeak: false,
        mood: 'helpful',
        hints: [
          'Unity can reach the backend',
          'Next step: replace mock reply with OpenClaw forwarding'
        ],
        error: null
      });
    });

    req.on('error', error => {
      return sendJson(res, 500, {
        success: false,
        sessionId: '',
        reply: '',
        shouldSpeak: false,
        mood: 'error',
        hints: [],
        error: error.message
      });
    });

    return;
  }

  return sendJson(res, 404, {
    success: false,
    error: 'Not found'
  });
});

server.listen(PORT, HOST, () => {
  console.log(`DuckXR backend listening on http://${HOST}:${PORT}`);
  console.log('POST /duck/chat');
  console.log('GET  /health');
});
