import { HumanMessage } from "@langchain/core/messages";
import type { DbClient } from "@agents/db";
import { searchMemories, incrementRetrievalCount } from "@agents/db";
import { generateEmbedding } from "../embeddings";
import type { GraphState } from "../state";

/**
 * Factory that creates the memory_injection graph node.
 *
 * The node runs once at the very start of each graph invocation
 * (before compaction and agent). It:
 *   1. Finds the latest HumanMessage in state (the current user input)
 *   2. Generates an embedding for it
 *   3. Retrieves the top-8 most semantically similar long-term memories
 *   4. Increments retrieval_count on the fetched rows
 *   5. Injects a [MEMORIA DEL USUARIO] block into systemPrompt
 *
 * If there are no memories or the embedding call fails, the node
 * returns the state unchanged so the agent can still run normally.
 */
export function createMemoryInjectionNode(params: {
  db: DbClient;
  userId: string;
}) {
  const { db, userId } = params;

  return async function memoryInjectionNode(
    state: typeof GraphState.State
  ): Promise<Partial<typeof GraphState.State>> {
    // Find the latest human message — that is the current user input
    const latestHuman = [...state.messages]
      .reverse()
      .find((m) => m instanceof HumanMessage);

    if (!latestHuman) return {};

    const userInput =
      typeof latestHuman.content === "string"
        ? latestHuman.content
        : JSON.stringify(latestHuman.content);

    try {
      const embedding = await generateEmbedding(userInput);
      const memories = await searchMemories(db, { userId, embedding, limit: 8 });

      if (memories.length === 0) return {};

      // Fire-and-forget: don't block the agent if the update fails
      incrementRetrievalCount(
        db,
        memories.map((m) => m.id)
      ).catch((err) => console.error("[memory_injection] increment failed:", err));

      const memoryBlock = [
        "[MEMORIA DEL USUARIO]",
        "Los siguientes recuerdos provienen de sesiones anteriores y son relevantes para la consulta actual:",
        "",
        ...memories.map(
          (m, i) =>
            `${i + 1}. [${m.type.toUpperCase()}] ${m.content}`
        ),
        "[/MEMORIA DEL USUARIO]",
      ].join("\n");

      const enrichedSystemPrompt = `${memoryBlock}\n\n${state.systemPrompt}`;

      return { systemPrompt: enrichedSystemPrompt };
    } catch (err) {
      // Memory injection is best-effort — never block the agent
      console.error("[memory_injection] failed, continuing without memories:", err);
      return {};
    }
  };
}
