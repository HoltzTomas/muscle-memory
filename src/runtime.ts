import {
  Output,
  embed as aiEmbed,
  generateText as aiGenerateText,
  type EmbeddingModel,
  type GenerateTextOnStepFinishCallback,
  type GenerateTextOnToolCallFinishCallback,
  type GenerateTextResult,
  type LanguageModel,
  type LanguageModelUsage,
  type Prompt,
  type ToolSet,
} from 'ai'
import type { z } from 'zod'

import { normalizeUsage } from './utils'

export type GenerateTextCall<TOOLS extends ToolSet = ToolSet> = Prompt & {
  model: LanguageModel
  tools?: TOOLS | undefined
  system?: string | undefined
  stopWhen?: unknown
  abortSignal?: AbortSignal | undefined
  headers?: Record<string, string> | undefined
  providerOptions?: Record<string, unknown> | undefined
  experimental_context?: unknown
  onStepFinish?: GenerateTextOnStepFinishCallback<TOOLS> | undefined
  onToolCallFinish?: GenerateTextOnToolCallFinishCallback<TOOLS> | undefined
}

export interface GenerateStructuredObjectCall<OBJECT> {
  model: LanguageModel
  schema: z.ZodType<OBJECT>
  prompt: string
  system?: string | undefined
  abortSignal?: AbortSignal | undefined
}

export interface RuntimeObjectResult<OBJECT> {
  object: OBJECT
  usage: ReturnType<typeof normalizeUsage>
  rawUsage?: LanguageModelUsage
}

export interface AiRuntime {
  embed(options: {
    model: EmbeddingModel
    value: string
    abortSignal?: AbortSignal | undefined
    headers?: Record<string, string> | undefined
    providerOptions?: Record<string, unknown> | undefined
  }): Promise<{ embedding: number[] }>
  generateText<TOOLS extends ToolSet>(
    options: GenerateTextCall<TOOLS>,
  ): Promise<GenerateTextResult<TOOLS, any>>
  generateObject<OBJECT>(
    options: GenerateStructuredObjectCall<OBJECT>,
  ): Promise<RuntimeObjectResult<OBJECT>>
}

export const defaultRuntime: AiRuntime = {
  async embed(options) {
    const result = await aiEmbed({
      model: options.model,
      value: options.value,
      ...(options.abortSignal ? { abortSignal: options.abortSignal } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
      ...(options.providerOptions
        ? { providerOptions: options.providerOptions as never }
        : {}),
    })
    return { embedding: result.embedding }
  },

  async generateText<TOOLS extends ToolSet>(
    options: GenerateTextCall<TOOLS>,
  ): Promise<GenerateTextResult<TOOLS, any>> {
    const { onToolCallFinish, onStepFinish, experimental_context, ...rest } = options

    return aiGenerateText({
      ...rest,
      ...(experimental_context ? { experimental_context } : {}),
      ...(onStepFinish ? { onStepFinish } : {}),
      ...(onToolCallFinish
        ? { experimental_onToolCallFinish: onToolCallFinish }
        : {}),
    } as any)
  },

  async generateObject<OBJECT>({
    model,
    schema,
    prompt,
    system,
    abortSignal,
  }: GenerateStructuredObjectCall<OBJECT>): Promise<RuntimeObjectResult<OBJECT>> {
    const result = await aiGenerateText({
      model,
      prompt,
      output: Output.object({
        schema,
      }),
      ...(system ? { system } : {}),
      ...(abortSignal ? { abortSignal } : {}),
    })

    return {
      object: result.output as OBJECT,
      usage: normalizeUsage(result.totalUsage),
      rawUsage: result.totalUsage,
    }
  },
}
