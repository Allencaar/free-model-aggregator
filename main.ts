// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// WHAT A TRIAL GIVES THIS SERVER
//   A brief at $MAKEFUN_PROMPT_FILE (the provider directory, the quality table, the workload
//   manifest) and an env at $MAKEFUN_ENV_URL serving the ten simulated providers under their own
//   base paths. The server probes each provider's catalog in the provider's own wire, builds one
//   catalog, listens on a free port, POSTs /register with that catalog, then answers the client's
//   twenty requests one at a time and polls GET /work every 100 ms, exiting 0 when it says done.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// HOW IT ROUTES (the score: quality 60, served 15, wire 10, failover 10, catalog 5)
//   - quality: every free model that meets the request's requirements is ranked by the quality
//     table's rating_milli, best first. Requirements are the manifest row for the request (the
//     client sends the manifest in order) united with what the body itself asks for: tools when
//     the body carries tools, vision when it carries an image, JSON mode when it carries an
//     OpenAI response_format. A paid model is never a candidate.
//   - wires: a request whose body asks for JSON mode can only ride a wire that carries it, so an
//     anthropic-wire provider is skipped for those. Same-wire calls forward the client's body
//     with only the model swapped (plus stream_options.include_usage on streamed OpenAI calls,
//     which otherwise report no usage); cross-wire calls go through lib/wire.ts's canonical form.
//   - failover: a refusal (402, 429, 5xx, or a failed fetch) is counted and the request moves to
//     the next candidate, best first, until one answers. Calls are strictly sequential: a
//     provider serves one call at a time, and two successful calls would fail the request, so
//     nothing is ever speculated in parallel.
//   - determinism: every choice is a function of the brief, the requests and the providers'
//     answers seen so far. No wall clock, no randomness.
//
// FILES
//   lib/providers.ts   the brief's documents, the catalog probe, the catalog report.
//   lib/wire.ts        the three wires' translations, for requests and for answers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  answerToAnthropic,
  answerToOpenAI,
  anthropicError,
  anthropicModelList,
  canonFromAnthropic,
  canonFromOpenAI,
  canonToAnthropic,
  canonToGemini,
  canonToOpenAI,
  hasImage,
  lastUserText,
  openAIError,
  openAIModelList,
  usageFromEcho,
  type CanonAnswer,
  type CanonRequest,
  type ClientWire,
  type ListedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the request fails over to the next candidate. */
const CALL_TIMEOUT_MS = 12000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Client API requests are served one at a time, in arrival order: the manifest's row by index. */
let manifestIndex = 0;
/** Serialize request handling so provider calls never overlap. */
let queue: Promise<void> = Promise.resolve();

/** Per-provider call state, learned from the providers' own answers. */
interface ProviderState {
  calls: number;
  refusals: number;
}
const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ListedModel[] {
  const out: ListedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      out.push({
        id: model.id,
        provider: probe.provider.id,
        display_name: model.display_name,
        context_length: caps.context_length,
        tools: caps.tools,
        vision: caps.vision,
        json: caps.json,
      });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  dirIndex: number;
}

/**
 * Every legal pick for one request, best first: free models meeting the requirements, on
 * providers that answered their probe. `jsonMode` in the body also rules out the anthropic
 * wire, which cannot carry it. Ties break toward the provider with fewer calls so far, then
 * the directory's order, then the model id: fully deterministic.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  brief.directory.providers.forEach((entry, dirIndex) => {
    const probe = probes.find((p) => p.provider.id === entry.id);
    if (probe === undefined || !probe.ok) return;
    if (jsonMode && entry.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, entry.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, dirIndex });
    }
  });
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      stateOf(a.probe.provider.id).calls - stateOf(b.probe.provider.id).calls ||
      a.dirIndex - b.dirIndex ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row united with the body's own. */
function requirementsFor(canon: CanonRequest): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = { ...(entry?.requires ?? {}) };
  if (canon.tools.length > 0) requires.tools = true;
  if (hasImage(canon)) requires.vision = true;
  if (canon.jsonMode) requires.json = true;
  return requires;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendBody(res, status, 'application/json', JSON.stringify(body));
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream URL and body for one candidate, in the provider's own wire. */
function upstreamCall(
  candidate: Candidate,
  clientWire: ClientWire,
  clientBody: Json,
  canon: CanonRequest,
): { url: string; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  if (provider.wire === 'gemini') {
    const action = canon.stream ? 'streamGenerateContent' : 'generateContent';
    return { url: `${base}/v1beta/models/${model}:${action}`, body: canonToGemini(canon, model) };
  }
  if (provider.wire === clientWire) {
    // Same wire: forward the client's body with only the model swapped, so every field the
    // provider echoes is the client's own. A streamed OpenAI call also asks for usage in the
    // stream, which the client's own body does not.
    const body: Json = { ...clientBody, model };
    if (provider.wire === 'openai' && canon.stream) {
      const options = isRecord(clientBody.stream_options) ? clientBody.stream_options : {};
      body.stream_options = { ...options, include_usage: true };
    }
    return { url: `${base}${provider.wire === 'openai' ? '/v1/chat/completions' : '/v1/messages'}`, body };
  }
  if (provider.wire === 'openai') return { url: `${base}/v1/chat/completions`, body: canonToOpenAI(canon, model) };
  return { url: `${base}/v1/messages`, body: canonToAnthropic(canon, model) };
}

/** The headers a provider call carries: its own auth scheme, in its own wire. */
function upstreamHeaders(provider: ProviderDirectoryEntry, clientReq: IncomingMessage, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') {
    const version = clientReq.headers['anthropic-version'];
    headers['anthropic-version'] = typeof version === 'string' ? version : '2023-06-01';
  }
  if (stream) headers.accept = 'text/event-stream';
  return headers;
}

type Attempt =
  | { ok: true }
  | { ok: false; status: number; detail: string };

/**
 * One candidate tried for one request. On success the client has been answered (same-wire calls
 * pipe the provider's answer through; cross-wire calls translate it whole) and true comes back.
 * On any refusal the provider's state records it and the caller fails over to the next candidate.
 */
async function tryCandidate(
  candidate: Candidate,
  clientWire: ClientWire,
  clientBody: Json,
  canon: CanonRequest,
  clientReq: IncomingMessage,
  res: ServerResponse,
): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const { url, body } = upstreamCall(candidate, clientWire, clientBody, canon);
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers: upstreamHeaders(provider, clientReq, canon.stream),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    stateOf(provider.id).refusals += 1;
    return { ok: false, status: 0, detail: (err as Error).message };
  }
  stateOf(provider.id).calls += 1;
  if (!upstream.ok) {
    stateOf(provider.id).refusals += 1;
    const detail = (await upstream.text()).slice(0, 300);
    return { ok: false, status: upstream.status, detail: `HTTP ${upstream.status}: ${detail}` };
  }
  if (provider.wire === clientWire) {
    // Same wire: the provider's answer is already in the client's shape, usage included.
    const contentType = upstream.headers.get('content-type') ?? (canon.stream ? 'text/event-stream' : 'application/json');
    res.writeHead(200, { 'content-type': contentType, 'cache-control': 'no-cache' });
    if (upstream.body !== null) {
      const reader = upstream.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
      }
    }
    res.end();
    return { ok: true };
  }
  const text = await upstream.text();
  const answer: CanonAnswer =
    provider.wire === 'openai'
      ? answerFromOpenAI(text, canon.stream)
      : provider.wire === 'anthropic'
        ? answerFromAnthropic(text, canon.stream)
        : answerFromGemini(text, canon.stream);
  if (answer.inputTokens === null || answer.outputTokens === null) {
    const computed = usageFromEcho(canon.system, lastUserText(canon), canon.tools.map((t) => t.name), answer.text);
    answer.inputTokens ??= computed.input;
    answer.outputTokens ??= computed.output;
  }
  const rendered =
    clientWire === 'openai' ? answerToOpenAI(answer, candidate.model.id, canon.stream) : answerToAnthropic(answer, candidate.model.id, canon.stream);
  sendBody(res, 200, rendered.contentType, rendered.body);
  return { ok: true };
}

/** Serve one client request (either wire) through the candidate list until one answers. */
async function serve(clientWire: ClientWire, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const canon = clientWire === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const candidates = candidatesFor(requirementsFor(canon), canon.jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  let last: Attempt = { ok: false, status: 0, detail: 'no candidate was tried' };
  for (const candidate of candidates) {
    const attempt = await tryCandidate(candidate, clientWire, body, canon, req, res);
    if (attempt.ok) return;
    last = attempt;
    log(`request ${manifestIndex}: ${candidate.probe.provider.id}/${candidate.model.id} refused (${attempt.detail})`);
  }
  sendError(res, clientWire, 503, 'all_providers_refused', `every free model that meets this request refused; last: ${last.detail}`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte; wait for stdout to flush.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    const run = queue.then(() => handle(req, res));
    queue = run.catch(() => undefined);
    void run.catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's own port on one layout; the ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
