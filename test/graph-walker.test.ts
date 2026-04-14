import { tool } from 'ai'
import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { GraphWalker } from '../src/graph-walker'
import type { Template } from '../src/types'

describe('GraphWalker', () => {
  it('resolves chained refs and follows failure edges', async () => {
    const calls: string[] = []

    const getOrder = tool({
      description: 'Fetch an order.',
      inputSchema: z.object({ id: z.string() }),
      execute: async ({ id }) => {
        calls.push(`getOrder:${id}`)
        return { email: 'person@example.com', total: 42 }
      },
    })

    const processRefund = tool({
      description: 'Process a refund.',
      inputSchema: z.object({
        email: z.string(),
        amount: z.number(),
      }),
      execute: async (): Promise<{ refundId: string }> => {
        calls.push('processRefund')
        throw new Error('Refund provider unavailable')
      },
    })

    const createCredit = tool({
      description: 'Create store credit.',
      inputSchema: z.object({
        email: z.string(),
        amount: z.number(),
      }),
      execute: async ({
        email,
        amount,
      }): Promise<{ creditId: string }> => {
        calls.push('createCredit')
        return { creditId: `${email}:${amount}` }
      },
    })

    const walker = new GraphWalker({ getOrder, processRefund, createCredit })
    const template: Template = {
      id: 'refund-fallback-v1',
      templateFamilyId: 'refund-fallback',
      version: 1,
      status: 'active',
      centroid: [1, 0, 0],
      keywords: ['refund'],
      confidence: 0.96,
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
      executionCount: 0,
      successRate: 1,
      traceCount: 5,
      args: {
        order_id: {
          type: 'extract_from_input',
          name: 'order_id',
          pattern: 'ORD-\\d+',
          description: 'Order id.',
          valueType: 'string',
        },
      },
      graph: {
        entryNode: '1:getOrder',
        nodes: {
          '1:getOrder': {
            id: '1:getOrder',
            tool: 'getOrder',
            args: { id: '{input.order_id}' },
          },
          '2:processRefund': {
            id: '2:processRefund',
            tool: 'processRefund',
            args: {
              email: '{1:getOrder.output.email}',
              amount: '{1:getOrder.output.total}',
            },
          },
          '3:createCredit': {
            id: '3:createCredit',
            tool: 'createCredit',
            args: {
              email: '{1:getOrder.output.email}',
              amount: '{1:getOrder.output.total}',
            },
          },
        },
        edges: [
          { from: '1:getOrder', to: '2:processRefund', weight: 1 },
          { from: '2:processRefund', to: null, weight: 0.8 },
          { from: '2:processRefund', to: '3:createCredit', weight: 0.2, onFailure: true },
          { from: '3:createCredit', to: null, weight: 1 },
        ],
      },
      governance: {
        ttlDays: 30,
        decayPerWeek: 0.01,
        failureThreshold: 3,
        minSuccessRate: 0.7,
        consecutiveFailures: 0,
        recentExecutions: [],
      },
    }

    const result = await walker.execute({
      template,
      inputValues: { order_id: 'ORD-900' },
    })

    expect(calls).toEqual(['getOrder:ORD-900', 'processRefund', 'createCredit'])
    expect(result.outputsByNode['3:createCredit']).toEqual({
      creditId: 'person@example.com:42',
    })
    expect(result.steps.map(step => [step.tool, step.success])).toEqual([
      ['getOrder', true],
      ['processRefund', false],
      ['createCredit', true],
    ])
  })
})
