# Muscle Memory

Muscle Memory is a TypeScript library that wraps the Vercel AI SDK and learns from successful tool-calling runs over time.

It starts as a normal tool-using agent, records structured traces, learns reusable execution graphs offline, and eventually replays high-confidence workflows deterministically without calling the large reasoning model again.

## Status

Muscle Memory is published on npm and ready for early adopters who want to experiment with learned tool-execution flows on top of the Vercel AI SDK.

Current scope:

- Node.js 18+
- Non-streaming agent execution
- `MemoryStore` for local development and tests
- `RedisStore` for persistent deployments
- Offline learning via `learn(...)`

Not shipped yet:

- Postgres and SQLite stores
- Streaming APIs
- Built-in dashboard or CLI
- Production hardening for every edge case

## Why Use It

Many agent workloads repeat the same job thousands of times.

Without a memory layer, every request pays full LLM cost again. Muscle Memory helps you keep the Vercel AI SDK workflow you already use, while adding:

- Trace recording for tool-based runs
- Offline template learning
- Intent matching through embeddings
- Cheap argument extraction for repeated workflows
- Deterministic graph execution with fallback to full reasoning

## Installation

Install the package plus the AI SDK peer dependency and whatever provider package you want to use:

```bash
pnpm add muscle-memory ai @ai-sdk/openai zod
```

Notes:

- `ai` is a peer dependency because Muscle Memory is meant to sit on top of the AI SDK already used by your app.
- `zod` is used in the examples because AI SDK tools commonly use `inputSchema`.
- `RedisStore` is included in this package. You only need a Redis URL when you choose to use it.

## Quick Start

This example uses the current AI SDK OpenAI provider style:

```ts
import { openai } from '@ai-sdk/openai'
import { tool } from 'ai'
import { z } from 'zod'

import { MemoryStore, muscleMemory } from 'muscle-memory'

const getOrder = tool({
  description: 'Get order details by ID',
  inputSchema: z.object({
    id: z.string(),
  }),
  execute: async ({ id }) => {
    return {
      status: 'paid',
      email: 'customer@example.com',
      total: 59.99,
      id,
    }
  },
})

const cancelOrder = tool({
  description: 'Cancel an order',
  inputSchema: z.object({
    id: z.string(),
  }),
  execute: async ({ id }) => {
    return {
      cancelled: true,
      id,
    }
  },
})

const processRefund = tool({
  description: 'Process a refund',
  inputSchema: z.object({
    email: z.string(),
    amount: z.number(),
  }),
  execute: async ({ email, amount }) => {
    return {
      refundId: `refund:${email}:${amount}`,
    }
  },
})

const agent = muscleMemory({
  model: openai('gpt-4.1'),
  extractionModel: openai('gpt-4.1-nano'),
  embeddingModel: openai.textEmbeddingModel('text-embedding-3-small'),
  tools: {
    getOrder,
    cancelOrder,
    processRefund,
  },
  store: new MemoryStore(),
  system: 'You are a customer support agent.',
  maxSteps: 5,
})

const result = await agent.run({
  prompt: 'Please cancel order ORD-412',
})

console.log(result.phase)      // 1 or 3
console.log(result.templateId) // null until a learned template is active
console.log(result.text)
console.log(result.traceId)
console.log(result.usage)
```

## Learning Templates

Tracing happens during normal `agent.run(...)` calls.

Learning happens out of band through `learn(...)`, which you can run in a cron, queue worker, or manual script:

```ts
import { MemoryStore, learn } from 'muscle-memory'

const store = new MemoryStore()

const result = await learn({
  store,
  minTraces: 5,
  clusterThreshold: 0.82,
  confidenceThreshold: 0.9,
})

console.log(result)
```

## Inspecting Templates

Use the `inspect` helpers to list templates, inspect learned graphs, and manually invalidate one:

```ts
import { MemoryStore, inspect } from 'muscle-memory'

const store = new MemoryStore()

const templates = await inspect.listTemplates({ store })
const graph = await inspect.getTemplateGraph({
  store,
  templateId: 'cancel-order-abc123-v1',
})

await inspect.invalidateTemplate({
  store,
  templateId: 'cancel-order-abc123-v1',
  reason: 'Downstream API changed',
})
```

## Stores

### MemoryStore

`MemoryStore` is exported from the root package and is the easiest way to get started:

```ts
import { MemoryStore } from 'muscle-memory'

const store = new MemoryStore()
```

### RedisStore

`RedisStore` is available through a subpath export:

```ts
import { RedisStore } from 'muscle-memory/stores/redis'

const store = new RedisStore({
  url: process.env.REDIS_URL!,
})
```

Use Redis when you want traces and templates to persist across processes or deployments.

## Public API

Root exports:

- `muscleMemory`
- `MuscleMemoryAgent`
- `MemoryStore`
- `learn`
- `inspect`
- package types such as `MuscleMemoryConfig`, `RunResult`, `Trace`, and `Template`

Subpath exports:

- `muscle-memory/stores/memory`
- `muscle-memory/stores/redis`

## Current Limitations

- The runtime path is built around `generateText`, not streaming.
- The learning loop is intentionally simple in v1 and works best for high-repetition workflows.
- Fallback from Phase 3 to Phase 1 is implemented, but template governance is still early-stage.
- Postgres and SQLite stores are not included yet.
- The package is ready for early adopters, not a fully hardened production platform.

## License

MIT
