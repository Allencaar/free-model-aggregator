# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)

One server that speaks both the OpenAI Chat Completions wire and the Anthropic Messages wire,
and routes every request to the best free model that can serve it across a directory of
free-tier providers (OpenAI-, Anthropic- and Gemini-wire upstreams).

Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## How it works

- On startup the server reads its brief (`MAKEFUN_PROMPT_FILE`): the provider directory, the
  public quality table (ratings and capabilities per model) and the workload manifest.
- It probes each provider's catalog route in that provider's own wire, builds one catalog, and
  registers it with the environment (`MAKEFUN_ENV_URL`). `node main.ts --list` prints exactly
  that catalog document without starting the server.
- For each request it ranks every free model that meets the request's requirements (tools,
  vision, JSON mode, context length) by rating, speaks the chosen provider's wire — translating
  through a canonical form when the client's wire differs — and fails over to the next-best
  model when a provider refuses (402/429/5xx). Calls are strictly sequential and every routing
  decision is a deterministic function of the brief, the requests and the providers' answers.

## Layout

- `main.ts` — the server: brief, catalog, routes, routing, registration, the `/work` poll.
- `lib/providers.ts` — brief parsing, the catalog probe, the catalog report.
- `lib/wire.ts` — translations between the OpenAI, Anthropic and Gemini wires for requests and
  answers, JSON and streaming, including stop-reason and usage mapping.

## Running

Node 22, standard library only:

```
MAKEFUN_ENV_URL=http://localhost:8080 MAKEFUN_PROMPT_FILE=brief.txt node main.ts
node main.ts --list   # print the catalog and exit
```

Licensed under the MIT License.
