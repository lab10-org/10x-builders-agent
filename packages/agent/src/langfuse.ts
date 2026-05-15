import { CallbackHandler } from "@langfuse/langchain";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { NodeSDK } from "@opentelemetry/sdk-node";
import type { RunnableConfig } from "@langchain/core/runnables";

declare global {
  // Reuse the SDK across Next.js hot reloads and repeated agent invocations.
  // eslint-disable-next-line no-var
  var __agentsLangfuseSdk: NodeSDK | undefined;
}

interface LangfuseConfigInput {
  userId: string;
  sessionId: string;
  runName: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

function hasLangfuseCredentials(): boolean {
  return Boolean(process.env.LANGFUSE_PUBLIC_KEY && process.env.LANGFUSE_SECRET_KEY);
}

function ensureLangfuseTracingStarted(): void {
  if (!hasLangfuseCredentials() || globalThis.__agentsLangfuseSdk) return;

  globalThis.__agentsLangfuseSdk = new NodeSDK({
    spanProcessors: [new LangfuseSpanProcessor()],
  });
  globalThis.__agentsLangfuseSdk.start();
}

const TRACE_ATTR_MAX_LEN = 200;

/** Langfuse `propagateAttributes` requires string metadata values (≤200 chars each). */
function toPropagatedStringMetadata(
  metadata: Record<string, unknown>
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, val] of Object.entries(metadata)) {
    const serialized =
      typeof val === "string" ? val : JSON.stringify(val);
    const within =
      serialized.length <= TRACE_ATTR_MAX_LEN
        ? serialized
        : `${serialized.slice(0, TRACE_ATTR_MAX_LEN - 1)}…`;
    out[key] = within;
  }
  return out;
}

interface LangfuseRootTraceOptions<T> {
  userId: string;
  sessionId: string;
  traceName: string;
  input: unknown;
  tags?: string[];
  metadata?: Record<string, unknown>;
  execute: () => Promise<T>;
  summarizeResult: (result: T) => unknown;
}

/**
 * Wraps LangGraph invocation in an explicit OTEL/Langfuse root observation so each
 * conversational turn keeps a populated name plus input/output — even when the
 * LangChain callback root does not classify graph inputs (resume commands,
 * deserialized checkpoint messages, etc.).
 */
export async function withLangfuseRootTrace<T>(
  options: LangfuseRootTraceOptions<T>
): Promise<T> {
  if (!hasLangfuseCredentials()) return options.execute();

  ensureLangfuseTracingStarted();

  return propagateAttributes(
    {
      userId: options.userId,
      sessionId: options.sessionId,
      traceName: options.traceName,
      tags: options.tags,
      ...(options.metadata
        ? { metadata: toPropagatedStringMetadata(options.metadata) }
        : {}),
    },
    () =>
      startActiveObservation(
        options.traceName,
        async (obs) => {
          obs.update({ input: options.input });
          try {
            const result = await options.execute();
            obs.update({ output: options.summarizeResult(result) });
            return result;
          } catch (error) {
            obs.update({
              level: "ERROR",
              statusMessage:
                error instanceof Error ? error.message : String(error),
            });
            throw error;
          }
        },
        { endOnExit: true }
      )
  );
}

export function createLangfuseRunnableConfig({
  userId,
  sessionId,
  runName,
  tags = [],
  metadata = {},
}: LangfuseConfigInput): RunnableConfig {
  if (!hasLangfuseCredentials()) {
    return {
      runName,
      tags,
      metadata,
    };
  }

  ensureLangfuseTracingStarted();

  const langfuseHandler = new CallbackHandler({
    userId,
    sessionId,
    tags,
    traceMetadata: metadata,
  });

  return {
    callbacks: [langfuseHandler],
    runName,
    tags,
    metadata: {
      ...metadata,
      langfuseUserId: userId,
      langfuseSessionId: sessionId,
    },
  };
}
