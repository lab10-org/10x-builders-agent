const EMBEDDING_MODEL = "openai/text-embedding-3-small";
const EMBEDDING_DIMENSIONS = 1536;

interface OpenRouterEmbeddingResponse {
  data: Array<{ embedding: number[] }>;
}

/**
 * Generate a 1536-dim embedding vector for the given text using
 * text-embedding-3-small via the OpenRouter API.
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("Missing OPENROUTER_API_KEY");

  const response = await fetch("https://openrouter.ai/api/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://agents.local",
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: text,
      dimensions: EMBEDDING_DIMENSIONS,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenRouter embeddings API error ${response.status}: ${errorText}`
    );
  }

  const json = (await response.json()) as OpenRouterEmbeddingResponse;
  const embedding = json.data?.[0]?.embedding;

  if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `Unexpected embedding shape: expected ${EMBEDDING_DIMENSIONS} dims, got ${embedding?.length}`
    );
  }

  return embedding;
}
