import { tool } from 'ai'
import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { MuscleMemoryAgent } from '../src/agent'
import { learn } from '../src/learner'
import { MemoryStore } from '../src/stores/memory'
import { createSupportRuntime, type FakeRuntimeState } from './helpers'

describe('MuscleMemoryAgent', () => {
  it('learns from repeated phase 1 runs and routes later runs through phase 3', async () => {
    const store = new MemoryStore()
    const orders: Record<
      string,
      { status: string; email: string; total: number; cancelled?: boolean; refunded?: boolean }
    > = {
      'ORD-100': { status: 'paid', email: 'ord-100@example.com', total: 59.99 },
      'ORD-101': { status: 'paid', email: 'ord-101@example.com', total: 59.99 },
      'ORD-102': { status: 'paid', email: 'ord-102@example.com', total: 59.99 },
      'ORD-103': { status: 'paid', email: 'ord-103@example.com', total: 59.99 },
      'ORD-104': { status: 'paid', email: 'ord-104@example.com', total: 59.99 },
      'ORD-999': { status: 'paid', email: 'ord-999@example.com', total: 59.99 },
    }

    const getOrder = tool({
      description: 'Get order details',
      inputSchema: z.object({ id: z.string() }),
      execute: async ({ id }) => {
        const order = orders[id]
        if (order == null) {
          throw new Error(`Order ${id} was not found`)
        }

        return {
          status: order.status,
          email: order.email,
          total: order.total,
        }
      },
    })

    const cancelOrder = tool({
      description: 'Cancel an order',
      inputSchema: z.object({ id: z.string() }),
      execute: async ({ id }) => {
        const order = orders[id]
        if (order == null) {
          throw new Error(`Order ${id} was not found`)
        }

        order.cancelled = true
        return { cancelled: true }
      },
    })

    const processRefund = tool({
      description: 'Process a refund',
      inputSchema: z.object({ email: z.string(), amount: z.number() }),
      execute: async ({ email }) => {
        const order = Object.values(orders).find(candidate => candidate.email === email)
        if (order == null) {
          throw new Error(`No order found for ${email}`)
        }

        order.refunded = true
        return { refundId: `refund-${email}` }
      },
    })

    const state: FakeRuntimeState = {
      phase1Calls: 0,
      responseCalls: 0,
      objectCalls: 0,
    }

    const agent = new MuscleMemoryAgent(
      {
        model: 'mock/full-model' as never,
        extractionModel: 'mock/cheap-model' as never,
        embeddingModel: 'mock/text-embedding' as never,
        tools: { getOrder, cancelOrder, processRefund },
        store,
        thresholds: {
          similarity: 0.8,
          confidence: 0.6,
        },
      },
      {
        runtime: createSupportRuntime({ state, orders }),
      },
    )

    for (const orderId of ['ORD-100', 'ORD-101', 'ORD-102', 'ORD-103', 'ORD-104']) {
      const result = await agent.run({
        prompt: `Please cancel order ${orderId}`,
      })
      expect(result.phase).toBe(1)
      expect(result.templateId).toBeNull()
    }

    await learn({
      store,
      minTraces: 3,
      clusterThreshold: 0.8,
      confidenceThreshold: 0.6,
    })

    const templates = await store.getAllTemplates()
    expect(templates).toHaveLength(1)
    expect(templates[0]?.status).toBe('active')

    const result = await agent.run({
      prompt: 'Please cancel order ORD-999',
    })

    expect(result.phase).toBe(3)
    expect(result.templateId).toBeTruthy()
    expect(result.text).toContain('ORD-999')
    expect(orders['ORD-999']?.cancelled).toBe(true)
    expect(orders['ORD-999']?.refunded).toBe(true)
    expect(state.phase1Calls).toBe(5)
    expect(state.objectCalls).toBe(0)
    expect(state.responseCalls).toBe(1)
  })
})
