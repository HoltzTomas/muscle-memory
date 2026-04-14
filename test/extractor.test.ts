import { describe, expect, it, vi } from 'vitest'

import { Extractor } from '../src/extractor'
import type { AiRuntime } from '../src/runtime'
import type { Template } from '../src/types'

describe('Extractor', () => {
  it('uses regex matches without calling the model', async () => {
    const runtime: AiRuntime = {
      embed: vi.fn(),
      generateText: vi.fn(),
      generateObject: vi.fn(async () => {
        throw new Error('generateObject should not be called')
      }),
    }

    const extractor = new Extractor(runtime, 'mock-model' as never)
    const template: Template = {
      id: 'cancel-order-v1',
      templateFamilyId: 'cancel-order',
      version: 1,
      status: 'active',
      centroid: [1, 0, 0],
      keywords: ['cancel', 'order'],
      confidence: 0.95,
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
      executionCount: 0,
      successRate: 1,
      traceCount: 5,
      args: {
        getorder_id: {
          type: 'extract_from_input',
          name: 'getorder_id',
          pattern: 'ORD-\\d+',
          description: 'Order identifier.',
          valueType: 'string',
        },
      },
      graph: {
        entryNode: '1:getOrder',
        nodes: {
          '1:getOrder': { id: '1:getOrder', tool: 'getOrder', args: { id: '{input.getorder_id}' } },
        },
        edges: [{ from: '1:getOrder', to: null, weight: 1 }],
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

    const result = await extractor.extract({
      inputText: 'Please cancel order ORD-412 right away.',
      template,
    })

    expect(result.values).toEqual({ getorder_id: 'ORD-412' })
    expect(result.usage.totalTokens).toBe(0)
    expect(runtime.generateObject).not.toHaveBeenCalled()
  })
})
