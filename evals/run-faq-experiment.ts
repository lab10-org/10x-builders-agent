/**
 * Eval script: runs the `basic-faq` Langfuse dataset through the chat API
 * using the @langfuse/client experiment runner.
 *
 * Evaluators (run as code, always enforced):
 *   - is_helpful       — LLM judge: did the agent helpfully answer the question?
 *   - is_out_of_scope  — LLM judge: did the agent go beyond the FAQ scope?
 *   - keyword_match    — heuristic: fraction of expected keywords present
 *
 * Usage:
 *   cp .env.example .env   # fill in values
 *   npm run run-faq
 */

import { LangfuseClient, type ExperimentTask, type Evaluator } from "@langfuse/client";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { createClient } from "@supabase/supabase-js";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(__dirname, ".env") });

// ── Config ───────────────────────────────────────────────────────────────────

const APP_BASE_URL = (process.env.APP_BASE_URL ?? "http://localhost:3000").replace(/\/$/, "");
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";
const EVAL_USER_EMAIL = process.env.EVAL_USER_EMAIL ?? "";
const EVAL_USER_PASSWORD = process.env.EVAL_USER_PASSWORD ?? "";
const EVAL_AUTH_COOKIE = process.env.EVAL_AUTH_COOKIE ?? "";
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";

const DATASET_NAME = "basic-faq";
const JUDGE_MODEL = "google/gemini-2.0-flash-001";

// ── Guards ───────────────────────────────────────────────────────────────────

function assertEnv() {
  const required = [
    "NEXT_PUBLIC_SUPABASE_URL",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "OPENROUTER_API_KEY",
  ];

  if (!EVAL_AUTH_COOKIE && (!EVAL_USER_EMAIL || !EVAL_USER_PASSWORD)) {
    required.push("EVAL_USER_EMAIL", "EVAL_USER_PASSWORD");
  }

  const missing = required.filter((k) => !process.env[k]);
  if (missing.length) {
    console.error(`\n✗ Missing env vars: ${missing.join(", ")}`);
    console.error("  Copy .env.example → .env and fill in the values.\n");
    process.exit(1);
  }
}

// ── Auth ─────────────────────────────────────────────────────────────────────

let _cookieHeader: string | null = null;

async function getCookieHeader(): Promise<string> {
  if (_cookieHeader) return _cookieHeader;

  if (EVAL_AUTH_COOKIE) {
    console.log("Auth: using EVAL_AUTH_COOKIE from env\n");
    _cookieHeader = EVAL_AUTH_COOKIE;
    return _cookieHeader;
  }

  console.log(`Auth: signing in as ${EVAL_USER_EMAIL} ...`);
  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { data, error } = await supabase.auth.signInWithPassword({
    email: EVAL_USER_EMAIL,
    password: EVAL_USER_PASSWORD,
  });

  if (error || !data.session) {
    throw new Error(`Supabase sign-in failed: ${error?.message ?? "no session returned"}`);
  }

  const projectRef = new URL(SUPABASE_URL).hostname.split(".")[0];
  const cookieName = `sb-${projectRef}-auth-token`;
  const cookieValue = encodeURIComponent(JSON.stringify(data.session));
  console.log("Auth: OK\n");

  _cookieHeader = `${cookieName}=${cookieValue}`;
  return _cookieHeader;
}

// ── API helpers ──────────────────────────────────────────────────────────────

async function createSession(cookieHeader: string): Promise<string> {
  const res = await fetch(`${APP_BASE_URL}/api/sessions`, {
    method: "POST",
    headers: { Cookie: cookieHeader },
  });
  if (!res.ok) throw new Error(`POST /api/sessions → ${res.status}: ${await res.text()}`);
  const { session } = (await res.json()) as { session: { id: string } };
  return session.id;
}

async function sendMessage(
  sessionId: string,
  message: string,
  cookieHeader: string,
): Promise<string> {
  const res = await fetch(`${APP_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Cookie: cookieHeader },
    body: JSON.stringify({ message, sessionId }),
  });
  if (!res.ok) throw new Error(`POST /api/chat → ${res.status}: ${await res.text()}`);

  const data = (await res.json()) as {
    response: string | null;
    pendingConfirmation: { tool_call_id: string } | null;
  };

  if (data.pendingConfirmation) {
    await fetch(`${APP_BASE_URL}/api/chat/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Cookie: cookieHeader },
      body: JSON.stringify({ toolCallId: data.pendingConfirmation.tool_call_id, action: "reject" }),
    });
    return "[tool confirmation required — auto-rejected during eval]";
  }

  return data.response ?? "";
}

// ── LLM judge helper ─────────────────────────────────────────────────────────

interface JudgeResult {
  score: number;
  reasoning: string;
}

// Max chars sent to judge to avoid hitting context limits and malformed JSON
const MAX_OUTPUT_FOR_JUDGE = 1200;

async function callJudge(prompt: string): Promise<JudgeResult> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
    },
    body: JSON.stringify({
      model: JUDGE_MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
    }),
  });

  if (!res.ok) throw new Error(`OpenRouter judge call failed: ${res.status}`);
  const data = (await res.json()) as { choices: { message: { content: string } }[] };
  const raw = data.choices[0].message.content;

  // Extract first valid JSON object even if the model appended extra text
  const match = raw.match(/\{[\s\S]*?\}/);
  if (!match) throw new Error(`Judge returned non-JSON: ${raw.slice(0, 200)}`);
  return JSON.parse(match[0]) as JudgeResult;
}

function truncate(text: string): string {
  return text.length > MAX_OUTPUT_FOR_JUDGE
    ? `${text.slice(0, MAX_OUTPUT_FOR_JUDGE)}…`
    : text;
}

// ── Evaluators ───────────────────────────────────────────────────────────────

const isHelpful: Evaluator = async ({ input, output, expectedOutput }) => {
  const prompt = `Eres un evaluador de un asistente FAQ sobre LAB10, una comunidad de AI builders.

Pregunta del usuario: "${input}"
Respuesta del asistente: "${truncate(String(output))}"
${expectedOutput ? `Respuesta esperada de referencia: "${truncate(String(expectedOutput))}"` : ""}

Evalúa si la respuesta del asistente es útil y responde adecuadamente la pregunta.
Una respuesta es útil si: aborda directamente la pregunta, es factualmente correcta sobre LAB10, y es suficientemente completa.

Devuelve únicamente un JSON con este formato exacto:
{"score": 1, "reasoning": "explicación breve"}

Donde score = 1 si la respuesta ES útil, score = 0 si NO es útil.`;

  const result = await callJudge(prompt);
  return { name: "is_helpful", value: result.score, comment: result.reasoning };
};

const isOutOfScope: Evaluator = async ({ input, output }) => {
  const prompt = `Eres un evaluador de un asistente FAQ sobre LAB10, una comunidad de AI builders.

El asistente solo debe responder preguntas relacionadas con LAB10: su misión, cursos, equipo, tecnologías, comunidad, filosofía y recursos.

Pregunta del usuario: "${input}"
Respuesta del asistente: "${truncate(String(output))}"

Evalúa si la respuesta del asistente sale del alcance esperado del FAQ de LAB10.
Considera fuera de alcance: respuestas sobre temas no relacionados con LAB10, ejecución de código, búsquedas web, o acciones externas.

Devuelve únicamente un JSON con este formato exacto:
{"score": 1, "reasoning": "explicación breve"}

Donde score = 1 si la respuesta está FUERA de alcance, score = 0 si está DENTRO del alcance esperado.`;

  const result = await callJudge(prompt);
  return { name: "is_out_of_scope", value: result.score, comment: result.reasoning };
};

const keywordMatch: Evaluator = async ({ output, expectedOutput }) => {
  const expected = String(expectedOutput ?? "");
  const actual = String(output ?? "");
  const keywords = (expected.toLowerCase().match(/\b\w{5,}\b/g) ?? []);
  const score =
    keywords.length === 0
      ? actual.toLowerCase().trim() === expected.toLowerCase().trim() ? 1 : 0
      : keywords.filter((w) => actual.toLowerCase().includes(w)).length / keywords.length;
  return { name: "keyword_match", value: score };
};

// ── Task ─────────────────────────────────────────────────────────────────────

const chatTask: ExperimentTask = async (item) => {
  const question =
    typeof item.input === "string"
      ? item.input
      : ((item.input as Record<string, unknown>)?.user_input as string) ?? String(item.input);

  const cookie = await getCookieHeader();
  const sessionId = await createSession(cookie);
  return await sendMessage(sessionId, question, cookie);
};

// ── OTel setup ───────────────────────────────────────────────────────────────

// Required by @langfuse/client experiment runner to deliver traces to Langfuse
function startOtel(): NodeSDK {
  const sdk = new NodeSDK({ spanProcessors: [new LangfuseSpanProcessor()] });
  sdk.start();
  return sdk;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  assertEnv();
  const otel = startOtel();

  const langfuse = new LangfuseClient();

  console.log("══════════════════════════════════════════════");
  console.log(`  Dataset : ${DATASET_NAME}`);
  console.log(`  API     : ${APP_BASE_URL}`);
  console.log(`  Judge   : ${JUDGE_MODEL}`);
  console.log("══════════════════════════════════════════════\n");

  // Trigger auth before the experiment starts so errors surface early
  await getCookieHeader();

  const dataset = await langfuse.dataset.get(DATASET_NAME);

  const result = await dataset.runExperiment({
    name: `basic-faq-${new Date().toISOString()}`,
    task: chatTask,
    evaluators: [isHelpful, isOutOfScope, keywordMatch],
    maxConcurrency: 3,
  });

  console.log("\n" + (await result.format()));
  console.log(`\nView results → Langfuse > Datasets > ${DATASET_NAME} > Runs\n`);

  await otel.shutdown();
}

main().catch((err) => {
  console.error("\n✗ Fatal error:", err);
  process.exit(1);
});
