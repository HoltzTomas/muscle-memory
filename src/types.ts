import type {
  ContentPart,
  EmbeddingModel,
  FinishReason,
  GenerateTextResult,
  LanguageModel,
  LanguageModelRequestMetadata,
  LanguageModelResponseMetadata,
  LanguageModelUsage,
  ModelMessage,
  Prompt,
  ProviderMetadata,
  ReasoningOutput,
  StaticToolCall,
  StaticToolResult,
  StepResult,
  StopCondition,
  ToolSet,
  TypedToolCall,
  TypedToolResult,
  Warning,
} from 'ai'

import type { Store } from './stores/interface'

export type MithrilPhase = 1 | 3

export interface ThresholdConfig {
  similarity?: number
  confidence?: number
}

export interface GovernanceConfig {
  ttlDays?: number
  decayPerWeek?: number
  failureThreshold?: number
  minSuccessRate?: number
  recentWindowSize?: number
}

export interface MithrilConfig<TOOLS extends ToolSet = ToolSet> {
  model: LanguageModel
  extractionModel: LanguageModel
  embeddingModel: EmbeddingModel
  tools: TOOLS
  store: Store
  system?: string
  maxSteps?: number
  thresholds?: ThresholdConfig
  governance?: GovernanceConfig
}

export type RunInput<TOOLS extends ToolSet = ToolSet> = Prompt & {
  system?: string | undefined
  maxSteps?: number
  stopWhen?: StopCondition<TOOLS> | Array<StopCondition<TOOLS>> | undefined
  abortSignal?: AbortSignal | undefined
  headers?: Record<string, string> | undefined
  providerOptions?: Record<string, unknown> | undefined
  experimental_context?: unknown
}

export interface NormalizedUsage {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  reasoningTokens?: number | undefined
  cachedInputTokens?: number | undefined
}

export interface SerializedError {
  name: string
  message: string
  stack?: string | undefined
  cause?: unknown
}

export interface TraceStep {
  id: string
  sequence: number
  stepNumber: number
  tool: string
  toolCallId: string
  args: Record<string, unknown>
  output?: unknown
  latency: number
  success: boolean
  error?: SerializedError | undefined
}

export interface Trace {
  id: string
  inputText: string
  inputEmbedding: number[]
  embeddingModel: string
  timestamp: string
  success: boolean
  processed: boolean
  phase: MithrilPhase
  templateId: string | null
  fallbackFromTemplateId?: string | null | undefined
  totalLatency: number
  responseText?: string | undefined
  finishReason?: string | undefined
  steps: TraceStep[]
  error?: SerializedError | undefined
}

export interface TemplateGovernanceState {
  ttlDays: number
  decayPerWeek: number
  failureThreshold: number
  minSuccessRate: number
  consecutiveFailures: number
  degradedReason?: string | undefined
  recentExecutions: Array<{
    success: boolean
    timestamp: string
  }>
}

export type ArgValueType = 'string' | 'number' | 'boolean'

export type ArgDefinition =
  | { type: 'constant'; value: unknown }
  | {
      type: 'extract_from_input'
      name: string
      pattern?: string | undefined
      description?: string | undefined
      valueType: ArgValueType
    }
  | { type: 'chained'; ref: string }
  | { type: 'unresolved' }

export type GraphArgumentValue = unknown

export interface GraphNode {
  id: string
  tool: string
  args: Record<string, GraphArgumentValue>
}

export interface GraphEdge {
  from: string
  to: string | null
  weight: number
  onFailure?: boolean | undefined
  condition?: {
    field: string
    equals?: unknown
    notEquals?: unknown
  }
}

export interface ExecutionGraph {
  entryNode: string
  nodes: Record<string, GraphNode>
  edges: GraphEdge[]
}

export interface Template {
  id: string
  templateFamilyId: string
  version: number
  status: 'learning' | 'active' | 'degraded'
  centroid: number[]
  keywords: string[]
  confidence: number
  createdAt: string
  lastUsedAt: string | null
  executionCount: number
  successRate: number
  traceCount: number
  args: Record<string, ArgDefinition>
  graph: ExecutionGraph
  governance: TemplateGovernanceState
}

export interface RunResult<TOOLS extends ToolSet = ToolSet> {
  content: Array<ContentPart<TOOLS>>
  text: string
  reasoning: Array<ReasoningOutput>
  reasoningText?: string | undefined
  files: Array<unknown>
  sources: Array<unknown>
  toolCalls: Array<TypedToolCall<TOOLS>>
  staticToolCalls: Array<TypedToolCall<TOOLS>>
  toolResults: Array<TypedToolResult<TOOLS>>
  staticToolResults: Array<TypedToolResult<TOOLS>>
  finishReason: FinishReason
  rawFinishReason?: string | undefined
  usage: NormalizedUsage
  totalUsage: NormalizedUsage
  warnings?: Warning[] | undefined
  request?: LanguageModelRequestMetadata | undefined
  response?:
    | (LanguageModelResponseMetadata & {
        messages?: unknown[] | undefined
        body?: unknown
      })
    | undefined
  providerMetadata?: ProviderMetadata | undefined
  phase: MithrilPhase
  templateId: string | null
  traceId: string
  latency: number
  steps: TraceStep[]
  trace: Trace
  rawResult?: GenerateTextResult<TOOLS, any> | undefined
  rawUsage?: LanguageModelUsage | undefined
  rawTotalUsage?: LanguageModelUsage | undefined
}

export interface StoreMetadata {
  embeddingModelId?: string
}

export interface LearnConfig {
  store: Store
  minTraces?: number
  clusterThreshold?: number
  confidenceThreshold?: number
}

export interface LearnResult {
  clustersFound: number
  templatesCreated: number
  templatesUpdated: number
  tracesProcessed: number
}

export interface MatchResult {
  template: Template
  similarity: number
}

export interface MatcherConfig {
  similarityThreshold: number
}

export interface ExtractedArguments {
  values: Record<string, unknown>
  usage: NormalizedUsage
  rawUsage?: LanguageModelUsage
}

export interface GraphWalkResult {
  steps: TraceStep[]
  outputsByNode: Record<string, unknown>
  terminalNodeId: string
}

export interface InspectListTemplatesOptions {
  store: Store
}

export interface InspectGetTemplateGraphOptions {
  store: Store
  templateId: string
}

export interface InspectInvalidateTemplateOptions {
  store: Store
  templateId: string
  reason: string
}

export interface MithrilEvents<TOOLS extends ToolSet = ToolSet> {
  'phase1:start': { traceId: string; input: string }
  'phase1:complete': {
    traceId: string
    steps: TraceStep[]
    latency: number
    usage: NormalizedUsage
  }
  'phase3:start': { traceId: string; templateId: string; input: string }
  'phase3:complete': {
    traceId: string
    templateId: string
    steps: TraceStep[]
    latency: number
    usage: NormalizedUsage
  }
  'phase3:fallback': { traceId: string; templateId: string; reason: string }
  'template:degraded': {
    templateId: string
    reason: string
    successRate: number
  }
  'template:activated': {
    templateId: string
    confidence: number
  }
}

export type StepSnapshot<TOOLS extends ToolSet = ToolSet> = StepResult<TOOLS>

export interface RunContext<TOOLS extends ToolSet = ToolSet> {
  input: RunInput<TOOLS>
  inputText: string
  traceId: string
  embedding: number[]
}

export type MaybePromise<T> = T | Promise<T>

export type MithrilMessage = ModelMessage
