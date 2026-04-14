import { describe, expect, it } from 'vitest'

import { learn } from '../src/learner'
import { MemoryStore } from '../src/stores/memory'
import type { Trace } from '../src/types'

describe('Learner', () => {
  it('builds an active template with extracted args and fallback edges', async () => {
    const store = new MemoryStore()
    const traces: Trace[] = [
      makeTrace('trace-1', 'Cancel order ORD-100', 'ORD-100', true),
      makeTrace('trace-2', 'Cancel order ORD-101', 'ORD-101', true),
      makeTrace('trace-3', 'Cancel order ORD-102', 'ORD-102', true),
      makeTrace('trace-4', 'Cancel order ORD-103', 'ORD-103', false),
      makeTrace('trace-5', 'Cancel order ORD-104', 'ORD-104', true),
    ]

    for (const trace of traces) {
      await store.saveTrace(trace)
    }

    const result = await learn({
      store,
      minTraces: 3,
      clusterThreshold: 0.8,
      confidenceThreshold: 0.6,
    })

    const templates = await store.getAllTemplates()
    expect(result.templatesCreated).toBe(1)
    expect(templates).toHaveLength(1)
    expect(templates[0]?.status).toBe('active')
    expect(templates[0]?.args.getorder_id).toMatchObject({
      type: 'extract_from_input',
      pattern: 'ORD-\\d+',
    })
    expect(templates[0]?.graph.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          from: '3:processRefund',
          to: '4:createCredit',
          onFailure: true,
        }),
      ]),
    )
  })
})

function makeTrace(
  id: string,
  inputText: string,
  orderId: string,
  refundSucceeds: boolean,
): Trace {
  const baseSteps = [
    {
      id: `${id}-step-1`,
      sequence: 0,
      stepNumber: 0,
      tool: 'getOrder',
      toolCallId: `${id}-tool-1`,
      args: { id: orderId },
      output: { email: `${orderId.toLowerCase()}@example.com`, total: 59.99 },
      latency: 5,
      success: true,
    },
    {
      id: `${id}-step-2`,
      sequence: 1,
      stepNumber: 1,
      tool: 'cancelOrder',
      toolCallId: `${id}-tool-2`,
      args: { id: orderId },
      output: { cancelled: true },
      latency: 4,
      success: true,
    },
    {
      id: `${id}-step-3`,
      sequence: 2,
      stepNumber: 2,
      tool: 'processRefund',
      toolCallId: `${id}-tool-3`,
      args: {
        email: `${orderId.toLowerCase()}@example.com`,
        amount: 59.99,
      },
      output: refundSucceeds ? { refundId: `refund-${orderId}` } : undefined,
      latency: 6,
      success: refundSucceeds,
      error: refundSucceeds
        ? undefined
        : { name: 'Error', message: 'Refund failed' },
    },
  ]

  const steps = refundSucceeds
    ? baseSteps
    : [
        ...baseSteps,
        {
          id: `${id}-step-4`,
          sequence: 3,
          stepNumber: 3,
          tool: 'createCredit',
          toolCallId: `${id}-tool-4`,
          args: {
            email: `${orderId.toLowerCase()}@example.com`,
            amount: 59.99,
          },
          output: { creditId: `credit-${orderId}` },
          latency: 3,
          success: true,
        },
      ]

  return {
    id,
    inputText,
    inputEmbedding: [1, 0, 0],
    embeddingModel: 'mock:text-embedding',
    timestamp: new Date().toISOString(),
    success: true,
    processed: false,
    phase: 1,
    templateId: null,
    totalLatency: 18,
    responseText: `Handled ${orderId}`,
    finishReason: 'stop',
    steps,
  }
}
