import type {
  OnStepFinishEvent,
  OnToolCallFinishEvent,
  ToolSet,
} from 'ai'

import type { MithrilPhase, SerializedError, StepSnapshot, Trace } from './types'
import { createTraceStep, serializeError } from './utils'

export class TraceRecorder<TOOLS extends ToolSet = ToolSet> {
  private readonly steps: Trace['steps'] = []
  private readonly snapshots: Array<StepSnapshot<TOOLS>> = []

  readonly onStepFinish = (event: OnStepFinishEvent<TOOLS>): void => {
    this.snapshots.push(event)
  }

  readonly onToolCallFinish = (
    event: OnToolCallFinishEvent<TOOLS>,
  ): void => {
    this.steps.push(
      createTraceStep({
        sequence: this.steps.length,
        stepNumber: event.stepNumber ?? this.steps.length,
        tool: event.toolCall.toolName,
        toolCallId: event.toolCall.toolCallId,
        args: coerceRecord(event.toolCall.input),
        output: event.success ? event.output : undefined,
        latency: event.durationMs,
        success: event.success,
        error: event.success ? undefined : serializeError(event.error),
      }),
    )
  }

  getTraceSteps(): Trace['steps'] {
    return [...this.steps]
  }

  getSnapshots(): Array<StepSnapshot<TOOLS>> {
    return [...this.snapshots]
  }

  buildTrace(options: {
    id: string
    inputText: string
    inputEmbedding: number[]
    embeddingModel: string
    timestamp: string
    success: boolean
    phase: MithrilPhase
    templateId: string | null
    fallbackFromTemplateId?: string | null | undefined
    totalLatency: number
    responseText?: string | undefined
    finishReason?: string | undefined
    error?: SerializedError | undefined
  }): Trace {
    return {
      ...options,
      processed: false,
      steps: this.getTraceSteps(),
    }
  }
}

function coerceRecord(value: unknown): Record<string, unknown> {
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  return { value }
}
