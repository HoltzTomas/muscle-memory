import type { GenerateTextCall, AiRuntime } from '../src/runtime'
import type { NormalizedUsage } from '../src/types'
import type { ToolSet } from 'ai'

export interface FakeRuntimeState {
  phase1Calls: number
  responseCalls: number
  objectCalls: number
}

export function createUsage(inputTokens = 10, outputTokens = 5) {
  return {
    inputTokens,
    inputTokenDetails: {
      noCacheTokens: inputTokens,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    },
    outputTokens,
    outputTokenDetails: {
      textTokens: outputTokens,
      reasoningTokens: 0,
    },
    totalTokens: inputTokens + outputTokens,
  }
}

export function createNormalizedUsage(
  inputTokens = 10,
  outputTokens = 5,
): NormalizedUsage {
  return {
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens,
    reasoningTokens: 0,
    cachedInputTokens: 0,
  }
}

export function extractInputText(options: GenerateTextCall<any>): string {
  if ('prompt' in options && typeof options.prompt === 'string') {
    return options.prompt
  }

  if ('messages' in options && Array.isArray(options.messages)) {
    const last = options.messages[options.messages.length - 1]
    if (last?.role === 'user' && typeof last.content === 'string') {
      return last.content
    }
  }

  return ''
}

export function createTextResult<TOOLS extends ToolSet>(
  text: string,
  extras: Partial<Record<'toolCalls' | 'toolResults' | 'staticToolCalls' | 'staticToolResults', unknown[]>> = {},
) {
  const usage = createUsage()

  return {
    content: [{ type: 'text', text }],
    text,
    reasoning: [],
    reasoningText: undefined,
    files: [],
    sources: [],
    toolCalls: (extras.toolCalls ?? []) as any,
    staticToolCalls: (extras.staticToolCalls ?? extras.toolCalls ?? []) as any,
    dynamicToolCalls: [],
    toolResults: (extras.toolResults ?? []) as any,
    staticToolResults: (extras.staticToolResults ?? extras.toolResults ?? []) as any,
    dynamicToolResults: [],
    finishReason: 'stop' as const,
    rawFinishReason: 'stop',
    usage,
    totalUsage: usage,
    warnings: undefined,
    request: {} as any,
    response: { messages: [] } as any,
    providerMetadata: undefined,
    steps: [],
  } as any
}

export function createSupportRuntime(options: {
  state: FakeRuntimeState
  orders: Record<
    string,
    {
      status: string
      email: string
      total: number
    }
  >
}): AiRuntime {
  return {
    async embed({ value }) {
      if (value.toLowerCase().includes('cancel')) {
        return { embedding: [1, 0, 0] }
      }

      return { embedding: [0, 1, 0] }
    },

    async generateObject({ prompt }) {
      options.state.objectCalls += 1
      const orderId = prompt.match(/ORD-\d+/)?.[0] ?? 'ORD-000'

      return {
        object: {
          getorder_id: orderId,
        } as any,
        usage: createNormalizedUsage(4, 2),
        rawUsage: createUsage(4, 2),
      }
    },

    async generateText<TOOLS extends ToolSet>(call: GenerateTextCall<TOOLS>) {
      const inputText = extractInputText(call)

      if (
        'prompt' in call &&
        typeof call.prompt === 'string' &&
        call.prompt.startsWith('Write a short user-facing response')
      ) {
        options.state.responseCalls += 1
        const orderId = call.prompt.match(/ORD-\d+/)?.[0] ?? 'ORD-000'
        return createTextResult<TOOLS>(
          `Your order ${orderId} has been cancelled and a refund is being processed.`,
        )
      }

      options.state.phase1Calls += 1
      const orderId = inputText.match(/ORD-\d+/)?.[0] ?? 'ORD-000'
      const order = options.orders[orderId] ?? {
        status: 'paid',
        email: `${orderId.toLowerCase()}@example.com`,
        total: 59.99,
      }

      const toolCalls = [
        {
          type: 'tool-call',
          toolCallId: `toolcall-${orderId}-1`,
          toolName: 'getOrder',
          input: { id: orderId },
        },
        {
          type: 'tool-call',
          toolCallId: `toolcall-${orderId}-2`,
          toolName: 'cancelOrder',
          input: { id: orderId },
        },
        {
          type: 'tool-call',
          toolCallId: `toolcall-${orderId}-3`,
          toolName: 'processRefund',
          input: { email: order.email, amount: order.total },
        },
      ]

      const toolResults = [
        {
          type: 'tool-result',
          toolCallId: `toolcall-${orderId}-1`,
          toolName: 'getOrder',
          input: { id: orderId },
          output: order,
        },
        {
          type: 'tool-result',
          toolCallId: `toolcall-${orderId}-2`,
          toolName: 'cancelOrder',
          input: { id: orderId },
          output: { cancelled: true },
        },
        {
          type: 'tool-result',
          toolCallId: `toolcall-${orderId}-3`,
          toolName: 'processRefund',
          input: { email: order.email, amount: order.total },
          output: { refundId: `refund-${orderId}` },
        },
      ]

      for (const result of toolResults) {
        await call.onToolCallFinish?.({
          stepNumber: 0,
          model: { provider: 'mock', modelId: 'mock-model' },
          toolCall: toolCalls.find(toolCall => toolCall.toolCallId === result.toolCallId) as any,
          messages: [],
          abortSignal: call.abortSignal,
          durationMs: 5,
          functionId: undefined,
          metadata: undefined,
          experimental_context: call.experimental_context,
          success: true,
          output: result.output,
        } as any)
      }

      await call.onStepFinish?.({
        stepNumber: 0,
        model: { provider: 'mock', modelId: 'mock-model' },
        functionId: undefined,
        metadata: undefined,
        experimental_context: call.experimental_context,
        content: [],
        text: '',
        reasoning: [],
        reasoningText: undefined,
        files: [],
        sources: [],
        toolCalls,
        staticToolCalls: toolCalls,
        dynamicToolCalls: [],
        toolResults,
        staticToolResults: toolResults,
        dynamicToolResults: [],
        finishReason: 'stop',
        rawFinishReason: 'stop',
        usage: createUsage(),
        warnings: undefined,
        request: {} as any,
        response: { messages: [] } as any,
        providerMetadata: undefined,
      } as any)

      return createTextResult<TOOLS>(
        `Your order ${orderId} has been cancelled and a refund is being processed.`,
        {
          toolCalls,
          toolResults,
        },
      )
    },
  }
}
