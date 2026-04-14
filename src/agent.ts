import { EventEmitter } from 'node:events'
import type {
  FinishReason,
  GenerateTextResult,
  ToolSet,
} from 'ai'

import { Extractor } from './extractor'
import { GovernanceManager } from './governance'
import { GraphWalker } from './graph-walker'
import { Matcher } from './matcher'
import type { AiRuntime } from './runtime'
import { defaultRuntime } from './runtime'
import { TraceRecorder } from './trace-recorder'
import type {
  MithrilConfig,
  MithrilEvents,
  NormalizedUsage,
  RunInput,
  RunResult,
  Template,
  Trace,
} from './types'
import {
  buildStopConditions,
  createId,
  getInputText,
  getThresholds,
  mergeUsages,
  normalizeUsage,
  serializeError,
} from './utils'

type InternalDependencies = {
  runtime?: AiRuntime
}

export class MithrilAgent<TOOLS extends ToolSet = ToolSet> extends EventEmitter {
  private readonly runtime: AiRuntime
  private readonly governance: GovernanceManager
  private readonly matcher: Matcher
  private readonly extractor: Extractor
  private readonly graphWalker: GraphWalker<TOOLS>
  private readonly thresholds: Required<NonNullable<MithrilConfig['thresholds']>>

  constructor(
    private readonly config: MithrilConfig<TOOLS>,
    dependencies: InternalDependencies = {},
  ) {
    super()
    this.runtime = dependencies.runtime ?? defaultRuntime
    this.thresholds = getThresholds(config.thresholds)
    this.governance = new GovernanceManager(config.store, config.governance, (
      eventName,
      payload,
    ) => {
      this.emit(eventName, payload)
    })
    this.matcher = new Matcher(
      config.store,
      this.governance,
      this.thresholds.similarity,
      this.thresholds.confidence,
    )
    this.extractor = new Extractor(this.runtime, config.extractionModel)
    this.graphWalker = new GraphWalker(config.tools)
  }

  override on<K extends keyof MithrilEvents<TOOLS>>(
    eventName: K,
    listener: (payload: MithrilEvents<TOOLS>[K]) => void,
  ): this
  override on(eventName: string | symbol, listener: (...args: any[]) => void): this {
    return super.on(eventName, listener)
  }

  async run(input: RunInput<TOOLS>): Promise<RunResult<TOOLS>> {
    const startedAt = Date.now()
    const traceId = createId('trace')
    const inputText = getInputText(input)
    const timestamp = new Date().toISOString()

    await this.governance.ensureEmbeddingModelIdentity(this.config.embeddingModel)

    const embedding = await this.tryEmbed(inputText, input.abortSignal)

    if (embedding.length > 0) {
      const match = await this.matcher.match(inputText, embedding)
      if (match != null) {
        this.emit('phase3:start', {
          traceId,
          templateId: match.template.id,
          input: inputText,
        })

        try {
          return await this.runPhase3({
            template: match.template,
            input,
            inputText,
            traceId,
            timestamp,
            inputEmbedding: embedding,
            startedAt,
          })
        } catch (error) {
          await this.governance.recordExecution(match.template, false)
          this.emit('phase3:fallback', {
            traceId,
            templateId: match.template.id,
            reason: error instanceof Error ? error.message : 'Phase 3 failure',
          })

          return this.runPhase1({
            input,
            inputText,
            traceId,
            timestamp,
            inputEmbedding: embedding,
            startedAt,
            fallbackFromTemplateId: match.template.id,
          })
        }
      }
    }

    return this.runPhase1({
      input,
      inputText,
      traceId,
      timestamp,
      inputEmbedding: embedding,
      startedAt,
    })
  }

  private async runPhase1(options: {
    input: RunInput<TOOLS>
    inputText: string
    traceId: string
    timestamp: string
    inputEmbedding: number[]
    startedAt: number
    fallbackFromTemplateId?: string
  }): Promise<RunResult<TOOLS>> {
      this.emit('phase1:start', {
      traceId: options.traceId,
      input: options.inputText,
    })

    const recorder = new TraceRecorder<TOOLS>()

    try {
      const result = await this.runtime.generateText<TOOLS>({
        model: this.config.model,
        tools: this.config.tools,
        ...extractPrompt(options.input),
        stopWhen: buildStopConditions(
          options.input.maxSteps ?? this.config.maxSteps,
          options.input.stopWhen,
        ),
        onStepFinish: recorder.onStepFinish,
        onToolCallFinish: recorder.onToolCallFinish,
        ...(options.input.system ?? this.config.system
          ? { system: options.input.system ?? this.config.system }
          : {}),
        ...(options.input.abortSignal
          ? { abortSignal: options.input.abortSignal }
          : {}),
        ...(options.input.headers ? { headers: options.input.headers } : {}),
        ...(options.input.providerOptions
          ? { providerOptions: options.input.providerOptions }
          : {}),
        ...(options.input.experimental_context
          ? { experimental_context: options.input.experimental_context }
          : {}),
      })

      const trace = recorder.buildTrace({
        id: options.traceId,
        inputText: options.inputText,
        inputEmbedding: options.inputEmbedding,
        embeddingModel: (await this.config.store.getMetadata()).embeddingModelId ?? '',
        timestamp: options.timestamp,
        success: true,
        phase: 1,
        templateId: null,
        fallbackFromTemplateId: options.fallbackFromTemplateId,
        totalLatency: Date.now() - options.startedAt,
        responseText: result.text,
        finishReason: result.finishReason,
        ...(options.fallbackFromTemplateId
          ? { fallbackFromTemplateId: options.fallbackFromTemplateId }
          : {}),
      })

      await this.config.store.saveTrace(trace)

      const usage = normalizeUsage(result.usage)
      const totalUsage = normalizeUsage(result.totalUsage)

      this.emit('phase1:complete', {
        traceId: options.traceId,
        steps: trace.steps,
        latency: trace.totalLatency,
        usage: totalUsage,
      })

      return buildPhase1RunResult(result, trace, usage, totalUsage)
    } catch (error) {
      const trace = recorder.buildTrace({
        id: options.traceId,
        inputText: options.inputText,
        inputEmbedding: options.inputEmbedding,
        embeddingModel: (await this.config.store.getMetadata()).embeddingModelId ?? '',
        timestamp: options.timestamp,
        success: false,
        phase: 1,
        templateId: null,
        totalLatency: Date.now() - options.startedAt,
        error: serializeError(error),
        ...(options.fallbackFromTemplateId
          ? { fallbackFromTemplateId: options.fallbackFromTemplateId }
          : {}),
      })

      await this.config.store.saveTrace(trace)
      throw error
    }
  }

  private async runPhase3(options: {
    template: Template
    input: RunInput<TOOLS>
    inputText: string
    traceId: string
    timestamp: string
    inputEmbedding: number[]
    startedAt: number
  }): Promise<RunResult<TOOLS>> {
    const extracted = await this.extractor.extract({
      inputText: options.inputText,
      template: options.template,
      ...(options.input.abortSignal
        ? { abortSignal: options.input.abortSignal }
        : {}),
    })

    const walked = await this.graphWalker.execute({
      template: options.template,
      inputValues: extracted.values,
      messages: 'messages' in options.input ? options.input.messages : undefined,
      abortSignal: options.input.abortSignal,
      experimentalContext: options.input.experimental_context,
    })

    const { toolCalls, toolResults } =
      this.graphWalker.buildToolArtifacts(walked.steps)

    const response = await this.runtime.generateText({
      model: this.config.extractionModel,
      abortSignal: options.input.abortSignal,
      prompt: buildResponsePrompt(
        options.inputText,
        options.template,
        extracted.values,
        walked.outputsByNode,
      ),
    })

    const responseUsage = normalizeUsage(response.totalUsage)
    const totalUsage = mergeUsages(extracted.usage, responseUsage)
    const trace = buildPhase3Trace({
      traceId: options.traceId,
      timestamp: options.timestamp,
      templateId: options.template.id,
      inputText: options.inputText,
      inputEmbedding: options.inputEmbedding,
      embeddingModel: (await this.config.store.getMetadata()).embeddingModelId ?? '',
      responseText: response.text,
      steps: walked.steps,
      totalLatency: Date.now() - options.startedAt,
      finishReason: response.finishReason,
    })

    await this.config.store.saveTrace(trace)
    await this.governance.recordExecution(options.template, true)

    this.emit('phase3:complete', {
      traceId: options.traceId,
      templateId: options.template.id,
      steps: walked.steps,
      latency: trace.totalLatency,
      usage: totalUsage,
    })

    return {
      content: [{ type: 'text', text: response.text }] as any,
      text: response.text,
      reasoning: [],
      reasoningText: response.reasoningText,
      files: [],
      sources: [],
      toolCalls,
      staticToolCalls: toolCalls,
      toolResults,
      staticToolResults: toolResults,
      finishReason: (response.finishReason ?? 'stop') as FinishReason,
      rawFinishReason: response.rawFinishReason,
      usage: totalUsage,
      totalUsage,
      warnings: response.warnings,
      request: response.request,
      response: response.response,
      providerMetadata: response.providerMetadata,
      phase: 3,
      templateId: options.template.id,
      traceId: options.traceId,
      latency: trace.totalLatency,
      steps: walked.steps,
      trace,
      rawResult: response as unknown as GenerateTextResult<TOOLS, any>,
      rawUsage: response.totalUsage,
      rawTotalUsage: response.totalUsage,
    }
  }

  private async tryEmbed(inputText: string, abortSignal?: AbortSignal): Promise<number[]> {
    if (inputText.length === 0) {
      return []
    }

    try {
      const result = await this.runtime.embed({
        model: this.config.embeddingModel,
        value: inputText,
        abortSignal,
      })
      return result.embedding
    } catch {
      return []
    }
  }
}

function extractPrompt<TOOLS extends ToolSet>(input: RunInput<TOOLS>) {
  if ('prompt' in input) {
    return { prompt: input.prompt }
  }

  if ('messages' in input) {
    return { messages: input.messages }
  }

  return {}
}

function buildPhase1RunResult<TOOLS extends ToolSet>(
  result: GenerateTextResult<TOOLS, any>,
  trace: Trace,
  usage: NormalizedUsage,
  totalUsage: NormalizedUsage,
): RunResult<TOOLS> {
  return {
    content: result.content,
    text: result.text,
    reasoning: result.reasoning,
    reasoningText: result.reasoningText,
    files: result.files,
    sources: result.sources,
    toolCalls: result.toolCalls,
    staticToolCalls: result.staticToolCalls,
    toolResults: result.toolResults,
    staticToolResults: result.staticToolResults,
    finishReason: result.finishReason,
    rawFinishReason: result.rawFinishReason,
    usage,
    totalUsage,
    warnings: result.warnings,
    request: result.request,
    response: result.response,
    providerMetadata: result.providerMetadata,
    phase: 1,
    templateId: null,
    traceId: trace.id,
    latency: trace.totalLatency,
    steps: trace.steps,
    trace,
    rawResult: result,
    rawUsage: result.usage,
    rawTotalUsage: result.totalUsage,
  }
}

function buildPhase3Trace(options: {
  traceId: string
  timestamp: string
  templateId: string
  inputText: string
  inputEmbedding: number[]
  embeddingModel: string
  responseText: string
  steps: Trace['steps']
  totalLatency: number
  finishReason?: string
}): Trace {
  return {
    id: options.traceId,
    inputText: options.inputText,
    inputEmbedding: options.inputEmbedding,
    embeddingModel: options.embeddingModel,
    timestamp: options.timestamp,
    success: true,
    processed: false,
    phase: 3,
    templateId: options.templateId,
    totalLatency: options.totalLatency,
    responseText: options.responseText,
    finishReason: options.finishReason,
    steps: options.steps,
  }
}

function buildResponsePrompt(
  inputText: string,
  template: Template,
  extractedValues: Record<string, unknown>,
  outputsByNode: Record<string, unknown>,
): string {
  return [
    'Write a short user-facing response based on the completed tool execution.',
    `Original user request: ${inputText}`,
    `Template: ${template.id}`,
    `Extracted values: ${JSON.stringify(extractedValues)}`,
    `Execution results: ${JSON.stringify(outputsByNode)}`,
  ].join('\n')
}
