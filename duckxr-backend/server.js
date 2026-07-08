import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'node:path';

const app = express();
const port = Number(process.env.PORT || 3001);
const openClawUrl = process.env.OPENCLAW_URL || 'http://127.0.0.1:18789/v1/responses';
const openClawToken = process.env.OPENCLAW_TOKEN || '';
const openClawModel = process.env.OPENCLAW_MODEL || 'openclaw:main';
const questClientOrigin = process.env.QUEST_CLIENT_ORIGIN || '*';
const allowedProjectRoots = (process.env.ALLOWED_PROJECT_ROOTS || '')
  .split(';')
  .map(value => value.trim())
  .filter(Boolean);

app.use(cors({ origin: questClientOrigin === '*' ? true : questClientOrigin }));
app.use(express.json({ limit: '32kb' }));

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'openclaw-quest-relay' });
});

function ensureOpenClawToken(res) {
  if (!openClawToken) {
    res.status(500).json({ error: 'Server missing OPENCLAW_TOKEN.' });
    return false;
  }

  return true;
}

function getResponseText(parsed) {
  const output = Array.isArray(parsed?.output) ? parsed.output : [];

  for (const item of output) {
    const content = Array.isArray(item?.content) ? item.content : [];
    for (const part of content) {
      if (typeof part?.text === 'string' && part.text.trim()) {
        return part.text;
      }
    }

    if (typeof item?.text === 'string' && item.text.trim()) {
      return item.text;
    }
  }

  return '[No output text found]';
}

async function callOpenClaw(payload) {
  const upstreamResponse = await fetch(openClawUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openClawToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  const rawText = await upstreamResponse.text();

  if (!upstreamResponse.ok) {
    return {
      ok: false,
      status: upstreamResponse.status,
      body: {
        error: 'OpenClaw upstream request failed.',
        details: rawText
      }
    };
  }

  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    return {
      ok: false,
      status: 502,
      body: {
        error: 'OpenClaw returned non-JSON response.',
        details: rawText
      }
    };
  }

  return {
    ok: true,
    parsed,
    text: getResponseText(parsed)
  };
}

function normalizeProjectPath(projectPath) {
  if (typeof projectPath !== 'string' || !projectPath.trim()) {
    return null;
  }

  return path.normalize(projectPath.trim());
}

function isProjectPathAllowed(projectPath) {
  if (allowedProjectRoots.length === 0) {
    return true;
  }

  const normalizedProjectPath = normalizeProjectPath(projectPath);
  if (!normalizedProjectPath) {
    return false;
  }

  return allowedProjectRoots.some(root => {
    const normalizedRoot = path.normalize(root);
    return normalizedProjectPath.toLowerCase().startsWith(normalizedRoot.toLowerCase());
  });
}

app.post('/quest/ask', async (req, res) => {
  try {
    const input = typeof req.body?.input === 'string' ? req.body.input.trim() : '';
    const user = typeof req.body?.user === 'string' && req.body.user.trim()
      ? req.body.user.trim()
      : 'meta-quest-relay';

    if (!input) {
      return res.status(400).json({ error: 'Missing input text.' });
    }

    if (!ensureOpenClawToken(res)) {
      return;
    }

    const result = await callOpenClaw({
      model: openClawModel,
      user,
      input
    });

    if (!result.ok) {
      return res.status(result.status).json(result.body);
    }

    return res.json({
      ok: true,
      text: result.text,
      raw: result.parsed
    });
  } catch (error) {
    return res.status(500).json({
      error: 'Relay failure.',
      details: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post('/quest/execute', async (req, res) => {
  try {
    const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt.trim() : '';
    const projectPath = normalizeProjectPath(req.body?.projectPath);
    const projectName = typeof req.body?.projectName === 'string' ? req.body.projectName.trim() : '';
    const user = typeof req.body?.user === 'string' && req.body.user.trim()
      ? req.body.user.trim()
      : 'meta-quest-execute';

    if (!prompt) {
      return res.status(400).json({ error: 'Missing prompt.' });
    }

    if (!projectPath) {
      return res.status(400).json({ error: 'Missing projectPath.' });
    }

    if (!isProjectPathAllowed(projectPath)) {
      return res.status(403).json({
        error: 'projectPath is not allowed by relay policy.',
        projectPath
      });
    }

    if (!ensureOpenClawToken(res)) {
      return;
    }

    const instruction = [
      'You are editing a Unity project on disk.',
      '',
      'Rules:',
      `- Only read, write, or edit files inside this project path: ${projectPath}`,
      '- Prefer minimal, targeted changes.',
      '- Prefer Unity scripts under Assets/Scripts unless the user says otherwise.',
      '- Do not make destructive changes unless the user explicitly asks.',
      '- Inspect relevant files first when needed before changing them.',
      '- After completing the task, return a concise summary of files created or edited.',
      '',
      projectName ? `Project name: ${projectName}` : '',
      `Project path: ${projectPath}`,
      '',
      'Task:',
      prompt
    ].filter(Boolean).join('\n');

    const result = await callOpenClaw({
      model: openClawModel,
      user,
      input: instruction
    });

    if (!result.ok) {
      return res.status(result.status).json(result.body);
    }

    return res.json({
      ok: true,
      text: result.text,
      projectPath,
      projectName: projectName || null,
      raw: result.parsed
    });
  } catch (error) {
    return res.status(500).json({
      error: 'Relay failure.',
      details: error instanceof Error ? error.message : String(error)
    });
  }
});

app.listen(port, '0.0.0.0', () => {
  console.log(`OpenClaw Quest relay listening on port ${port}`);
});
