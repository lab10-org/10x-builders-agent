import { CallbackHandler } from "@langfuse/langchain";
import { LangfuseSpanProcessor } from "@langfuse/otel";
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
