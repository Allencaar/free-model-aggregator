// Translation between the three wires: OpenAI Chat Completions, Anthropic Messages and Gemini.
//
// The client speaks OpenAI or Anthropic; a provider speaks its own wire. Every request is first
// read into one canonical form, then written out in the provider's wire; every provider answer is
// read back into one canonical answer and written out in the client's wire. The canonical form is
// built so the fields a trial compares survive the round trip exactly:
//
//   - the system text, byte for byte (each wire's parser joins its blocks with "\n");
//   - the last user message's text, block by block;
//   - an image, as the same bytes (base64) or the same URL on every wire;
//   - the tool names, in order (definitions ride along so the call stays a real one);
//   - max_tokens and the stop sequences, as the same numbers and strings;
//   - JSON mode: an OpenAI response_format becomes Gemini's responseMimeType. The Anthropic wire
//     cannot carry it at all, which the router (main.ts) treats as a wire constraint.
//
// Answers keep the provider's text whole, its usage counts (mapped between the wires' names) and
// its stop reason, mapped between the wires' vocabularies by family: end_turn/STOP -> stop,
// max_tokens/MAX_TOKENS -> length, tool_use -> tool_calls, refusal/SAFETY -> content_filter, and
// the reverse into Anthropic's names. Streaming answers are buffered whole, then re-emitted in
// the client's event shape; that keeps usage placement and event order exact on every pair.

import { isRecord, type Json } from './providers.ts';

export type ClientWire = 'openai' | 'anthropic';

export interface CanonTool {
  name: string;
  description: string;
  parameters: unknown;
}

export type CanonPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string | null; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: unknown }
  | { kind: 'tool_result'; id: string; content: string };

export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: CanonPart[];
}

export interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  stream: boolean;
}

export interface CanonAnswer {
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
  stop: string | null;
}

/** A fixed `created` value for emitted answers: no wall clock anywhere in an answer. */
const FIXED_CREATED = 1767225600;
/** Streamed text is cut into pieces of this many characters, as the simulated providers cut it. */
const SSE_PIECE_CHARS = 13;

// --------------------------------------------------------------------------- reading client bodies

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** An OpenAI image_url value as a canonical image part: a data URL becomes its base64 payload. */
function openAIImagePart(url: unknown): CanonPart {
  if (typeof url === 'string') {
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] !== undefined) return { kind: 'image', mediaType: data[1] ?? null, data: data[3], url: null };
    return { kind: 'image', mediaType: null, data: null, url };
  }
  return { kind: 'image', mediaType: null, data: null, url: null };
}

function openAIMessageParts(message: Json): CanonPart[] {
  const parts: CanonPart[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ kind: 'text', text: content });
  } else if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url)) parts.push(openAIImagePart(block.image_url.url));
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
      let input: unknown = {};
      if (typeof call.function.arguments === 'string') {
        try {
          input = JSON.parse(call.function.arguments);
        } catch {
          input = {};
        }
      }
      parts.push({ kind: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: call.function.name, input });
    }
  }
  return parts;
}

/** An OpenAI Chat Completions body read into the canonical form. */
export function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role === 'system') {
      const text = blocksText(entry.content);
      if (text !== '') systemParts.push(text);
    } else if (entry.role === 'user' || entry.role === 'assistant') {
      messages.push({ role: entry.role, parts: openAIMessageParts(entry) });
    } else if (entry.role === 'tool') {
      const content = typeof entry.content === 'string' ? entry.content : blocksText(entry.content);
      messages.push({ role: 'tool', parts: [{ kind: 'tool_result', id: typeof entry.tool_call_id === 'string' ? entry.tool_call_id : '', content }] });
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: tool.function.parameters ?? { type: 'object' },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOrNull(body.temperature),
    stream: body.stream === true,
  };
}

function anthropicMessageParts(message: Json): CanonPart[] {
  const parts: CanonPart[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ kind: 'text', text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ kind: 'image', mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : null, data: block.source.data, url: null });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ kind: 'image', mediaType: null, data: null, url: block.source.url });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      parts.push({ kind: 'tool_use', id: typeof block.id === 'string' ? block.id : '', name: block.name, input: block.input ?? {} });
    } else if (block.type === 'tool_result') {
      parts.push({
        kind: 'tool_result',
        id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: typeof block.content === 'string' ? block.content : blocksText(block.content),
      });
    }
  }
  return parts;
}

/** An Anthropic Messages body read into the canonical form. Its jsonMode is always false. */
export function canonFromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role === 'user' || entry.role === 'assistant') messages.push({ role: entry.role, parts: anthropicMessageParts(entry) });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: tool.input_schema ?? { type: 'object' },
      });
    }
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
    stream: body.stream === true,
  };
}

/** The canonical request's last user text, exactly as a provider's parser would read it back. */
export function lastUserText(canon: CanonRequest): string {
  let text = '';
  for (const message of canon.messages) {
    if (message.role !== 'user') continue;
    text = message.parts
      .filter((p): p is Extract<CanonPart, { kind: 'text' }> => p.kind === 'text')
      .map((p) => p.text)
      .join('\n');
  }
  return text;
}

/** Whether the canonical request carries an image anywhere (a vision requirement in the body). */
export function hasImage(canon: CanonRequest): boolean {
  return canon.messages.some((m) => m.parts.some((p) => p.kind === 'image'));
}

// --------------------------------------------------------------------------- writing provider bodies

function openAIParts(parts: CanonPart[]): { content: unknown; toolCalls: Json[]; toolMessages: Json[] } {
  const content: unknown[] = [];
  const toolCalls: Json[] = [];
  const toolMessages: Json[] = [];
  let onlyText = parts.length > 0;
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') {
      onlyText = false;
      const url = part.data !== null ? `data:${part.mediaType ?? 'image/png'};base64,${part.data}` : (part.url ?? '');
      content.push({ type: 'image_url', image_url: { url } });
    } else if (part.kind === 'tool_use') {
      onlyText = false;
      toolCalls.push({
        id: part.id,
        type: 'function',
        function: { name: part.name, arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input ?? {}) },
      });
    } else {
      onlyText = false;
      toolMessages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
    }
  }
  const single = content.length === 1 && content[0] !== undefined && (content[0] as Json).type === 'text';
  return { content: onlyText && single ? (content[0] as Json).text : content, toolCalls, toolMessages };
}

/** The canonical request written as an OpenAI Chat Completions body for `model`. */
export function canonToOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      for (const part of message.parts) {
        if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
      }
      continue;
    }
    const rendered = openAIParts(message.parts);
    const out: Json = { role: message.role, content: rendered.content };
    if (rendered.toolCalls.length > 0) out.tool_calls = rendered.toolCalls;
    messages.push(out, ...rendered.toolMessages);
  }
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (canon.temperature !== null) body.temperature = canon.temperature;
  // Ask the provider to report usage inside its event stream: a streamed OpenAI answer carries
  // none otherwise, and the usage counts are part of the answer the client is owed.
  if (canon.stream) body.stream_options = { include_usage: true };
  return body;
}

function anthropicParts(parts: CanonPart[]): unknown[] {
  const content: unknown[] = [];
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') {
      content.push({
        type: 'image',
        source:
          part.data !== null
            ? { type: 'base64', media_type: part.mediaType ?? 'image/png', data: part.data }
            : { type: 'url', url: part.url ?? '' },
      });
    } else if (part.kind === 'tool_use') content.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input ?? {} });
    else content.push({ type: 'tool_result', tool_use_id: part.id, content: part.content });
  }
  return content;
}

/** The canonical request written as an Anthropic Messages body for `model`. */
export function canonToAnthropic(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of canon.messages) {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    messages.push({ role, content: anthropicParts(message.parts) });
  }
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.system !== '') body.system = canon.system;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  return body;
}

function geminiParts(parts: CanonPart[]): unknown[] {
  const out: unknown[] = [];
  for (const part of parts) {
    if (part.kind === 'text') out.push({ text: part.text });
    else if (part.kind === 'image') {
      out.push(
        part.data !== null
          ? { inlineData: { mimeType: part.mediaType ?? 'image/png', data: part.data } }
          : { fileData: { fileUri: part.url ?? '' } },
      );
    } else if (part.kind === 'tool_use') out.push({ functionCall: { name: part.name, args: part.input ?? {} } });
    else out.push({ functionResponse: { name: part.id, response: { result: part.content } } });
  }
  return out;
}

/** The canonical request written as a Gemini generateContent body for `model`. */
export function canonToGemini(canon: CanonRequest, model: string): Json {
  const contents: Json[] = [];
  for (const message of canon.messages) {
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts: geminiParts(message.parts) });
  }
  const body: Json = { contents };
  if (canon.system !== '') body.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    body.tools = [
      { functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// --------------------------------------------------------------------------- reading provider answers

function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI wire answer, JSON or SSE, read into the canonical answer. */
export function answerFromOpenAI(text: string, stream: boolean): CanonAnswer {
  if (!stream) {
    const body = parseJson(text);
    const choice = body !== null && Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    let content = '';
    if (message !== null) {
      if (typeof message.content === 'string') content = message.content;
      else if (Array.isArray(message.content)) {
        content = message.content
          .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
          .map((b) => b.text as string)
          .join('');
      }
    }
    const usage = body !== null && isRecord(body.usage) ? body.usage : null;
    return {
      text: content,
      inputTokens: usage !== null ? numberOrNull(usage.prompt_tokens) : null,
      outputTokens: usage !== null ? numberOrNull(usage.completion_tokens) : null,
      stop: choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  let content = '';
  let stop: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJson(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrNull(chunk.usage.prompt_tokens) ?? input;
      output = numberOrNull(chunk.usage.completion_tokens) ?? output;
    }
    if (!Array.isArray(chunk.choices) || !isRecord(chunk.choices[0])) continue;
    const choice = chunk.choices[0];
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') content += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: content, inputTokens: input, outputTokens: output, stop };
}

/** An Anthropic wire answer, JSON or SSE, read into the canonical answer. */
export function answerFromAnthropic(text: string, stream: boolean): CanonAnswer {
  if (!stream) {
    const body = parseJson(text);
    let content = '';
    if (body !== null && Array.isArray(body.content)) {
      content = body.content
        .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
        .map((b) => b.text as string)
        .join('');
    }
    const usage = body !== null && isRecord(body.usage) ? body.usage : null;
    return {
      text: content,
      inputTokens: usage !== null ? numberOrNull(usage.input_tokens) : null,
      outputTokens: usage !== null ? numberOrNull(usage.output_tokens) : null,
      stop: body !== null && typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  let content = '';
  let stop: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of sseFrames(text)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrNull(data.message.usage.input_tokens) ?? input;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
      if (typeof data.delta.text === 'string') content += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrNull(data.usage.output_tokens) ?? output;
    }
  }
  return { text: content, inputTokens: input, outputTokens: output, stop };
}

/** A Gemini wire answer, JSON or SSE, read into the canonical answer. */
export function answerFromGemini(text: string, stream: boolean): CanonAnswer {
  const fromBody = (body: Json | null): CanonAnswer => {
    let content = '';
    let stop: string | null = null;
    if (body !== null && Array.isArray(body.candidates) && isRecord(body.candidates[0])) {
      const candidate = body.candidates[0];
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        content = candidate.content.parts
          .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
          .map((p) => p.text as string)
          .join('');
      }
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    const usage = body !== null && isRecord(body.usageMetadata) ? body.usageMetadata : null;
    return {
      text: content,
      inputTokens: usage !== null ? numberOrNull(usage.promptTokenCount) : null,
      outputTokens: usage !== null ? numberOrNull(usage.candidatesTokenCount) : null,
      stop,
    };
  };
  if (!stream) return fromBody(parseJson(text));
  let content = '';
  let stop: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of sseFrames(text)) {
    const piece = fromBody(parseJson(frame.data));
    content += piece.text;
    if (piece.stop !== null) stop = piece.stop;
    if (piece.inputTokens !== null) input = piece.inputTokens;
    if (piece.outputTokens !== null) output = piece.outputTokens;
  }
  return { text: content, inputTokens: input, outputTokens: output, stop };
}

// --------------------------------------------------------------------------- writing client answers

const STOP_TO_OPENAI: Record<string, string> = {
  stop: 'stop',
  end_turn: 'stop',
  STOP: 'stop',
  length: 'length',
  max_tokens: 'length',
  MAX_TOKENS: 'length',
  tool_calls: 'tool_calls',
  tool_use: 'tool_calls',
  function_call: 'tool_calls',
  content_filter: 'content_filter',
  refusal: 'content_filter',
  SAFETY: 'content_filter',
};

const STOP_TO_ANTHROPIC: Record<string, string> = {
  stop: 'end_turn',
  end_turn: 'end_turn',
  STOP: 'end_turn',
  length: 'max_tokens',
  max_tokens: 'max_tokens',
  MAX_TOKENS: 'max_tokens',
  tool_calls: 'tool_use',
  tool_use: 'tool_use',
  function_call: 'tool_use',
  content_filter: 'refusal',
  refusal: 'refusal',
  SAFETY: 'refusal',
};

/** A stop reason mapped into the OpenAI vocabulary. */
export function stopForOpenAI(stop: string | null): string {
  return (stop !== null ? STOP_TO_OPENAI[stop] : undefined) ?? 'stop';
}

/** A stop reason mapped into the Anthropic vocabulary. */
export function stopForAnthropic(stop: string | null): string {
  return (stop !== null ? STOP_TO_ANTHROPIC[stop] : undefined) ?? 'end_turn';
}

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE_CHARS) pieces.push(text.slice(at, at + SSE_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The canonical answer written as an OpenAI completion or event stream. */
export function answerToOpenAI(answer: CanonAnswer, model: string, stream: boolean): { contentType: string; body: string } {
  const finish = stopForOpenAI(answer.stop);
  const usage = {
    prompt_tokens: answer.inputTokens ?? 0,
    completion_tokens: answer.outputTokens ?? 0,
    total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0),
  };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finish, logprobs: null }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let out = sse(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(answer.text)) out += sse(null, chunk({ content: piece }, null));
  out += sse(null, chunk({}, finish));
  out += sse(null, { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage });
  out += 'data: [DONE]\n\n';
  return { contentType: 'text/event-stream', body: out };
}

/** The canonical answer written as an Anthropic message or event stream. */
export function answerToAnthropic(answer: CanonAnswer, model: string, stream: boolean): { contentType: string; body: string } {
  const stopReason = stopForAnthropic(answer.stop);
  const input = answer.inputTokens ?? 0;
  const output = answer.outputTokens ?? 0;
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: output },
      }),
    };
  }
  let out = sse('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: input, output_tokens: 0 },
    },
  });
  out += sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitPieces(answer.text)) {
    out += sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sse('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sse('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } });
  out += sse('message_stop', { type: 'message_stop' });
  return { contentType: 'text/event-stream', body: out };
}

// --------------------------------------------------------------------------- errors and model lists

/** An error body in the OpenAI envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error body in the Anthropic envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export interface ListedModel {
  id: string;
  provider: string;
  display_name: string;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** GET /v1/models in the OpenAI shape: exactly the free models the catalog registered. */
export function openAIModelList(models: ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
      free: true,
      context_length: m.context_length,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    })),
  };
}

/** GET /v1/models in the Anthropic shape: the same free model ids under a `data` list. */
export function anthropicModelList(models: ListedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: '2026-01-01T00:00:00Z',
      free: true,
      context_length: m.context_length,
      tools: m.tools,
      vision: m.vision,
      json: m.json,
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/**
 * The usage a provider reports, recomputed from the request it was sent and the text it answered:
 * the fallback for a stream that carried no usage of its own. Mirrors the trial's own arithmetic.
 */
export function usageFromEcho(systemText: string, userText: string, toolNames: string[], answerText: string): { input: number; output: number } {
  const prompt = systemText.length + userText.length + toolNames.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}
