import { cosineSimilarity, stepCountIs, type LanguageModelUsage, type Prompt } from 'ai'
import { createHash, randomUUID } from 'node:crypto'

import type {
  ArgValueType,
  GraphEdge,
  GraphNode,
  GovernanceConfig,
  NormalizedUsage,
  SerializedError,
  Template,
  ThresholdConfig,
  TraceStep,
} from './types'

const DEFAULT_THRESHOLDS = {
  similarity: 0.85,
  confidence: 0.9,
} satisfies Required<ThresholdConfig>

const DEFAULT_GOVERNANCE = {
  ttlDays: 30,
  decayPerWeek: 0.01,
  failureThreshold: 3,
  minSuccessRate: 0.7,
  recentWindowSize: 20,
} satisfies Required<GovernanceConfig>

const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'i',
  'in',
  'is',
  'it',
  'me',
  'my',
  'of',
  'on',
  'or',
  'please',
  'the',
  'to',
  'want',
  'with',
  'you',
])

export function createId(prefix: string): string {
  return `${prefix}-${randomUUID()}`
}

export function stableStringify(value: unknown): string {
  return JSON.stringify(sortValue(value))
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortValue)
  }

  if (value != null && typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>)
      .sort()
      .reduce<Record<string, unknown>>((accumulator, key) => {
        accumulator[key] = sortValue((value as Record<string, unknown>)[key])
        return accumulator
      }, {})
  }

  return value
}

export function deepEqual(left: unknown, right: unknown): boolean {
  return stableStringify(left) === stableStringify(right)
}

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value))
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value))
}

export function normalizeUsage(usage?: LanguageModelUsage): NormalizedUsage {
  return {
    inputTokens: usage?.inputTokens ?? 0,
    outputTokens: usage?.outputTokens ?? 0,
    totalTokens: usage?.totalTokens ?? 0,
    reasoningTokens:
      usage?.outputTokenDetails.reasoningTokens ?? usage?.reasoningTokens,
    cachedInputTokens:
      usage?.inputTokenDetails.cacheReadTokens ?? usage?.cachedInputTokens,
  }
}

export function mergeUsages(
  ...usages: Array<NormalizedUsage | undefined>
): NormalizedUsage {
  return usages.reduce<NormalizedUsage>(
    (accumulator, usage) => ({
      inputTokens: accumulator.inputTokens + (usage?.inputTokens ?? 0),
      outputTokens: accumulator.outputTokens + (usage?.outputTokens ?? 0),
      totalTokens: accumulator.totalTokens + (usage?.totalTokens ?? 0),
      reasoningTokens:
        (accumulator.reasoningTokens ?? 0) + (usage?.reasoningTokens ?? 0),
      cachedInputTokens:
        (accumulator.cachedInputTokens ?? 0) + (usage?.cachedInputTokens ?? 0),
    }),
    { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
  )
}

export function serializeError(error: unknown): SerializedError {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      cause: error.cause,
    }
  }

  return {
    name: 'Error',
    message: typeof error === 'string' ? error : 'Unknown error',
    cause: error,
  }
}

export function getEmbeddingModelIdentity(model: {
  provider?: string
  modelId?: string
} | string): string {
  if (typeof model === 'string') {
    return model
  }

  return [model.provider ?? 'unknown-provider', model.modelId ?? 'unknown-model']
    .join(':')
}

export function getThresholds(
  thresholds?: ThresholdConfig,
): Required<ThresholdConfig> {
  return {
    similarity: thresholds?.similarity ?? DEFAULT_THRESHOLDS.similarity,
    confidence: thresholds?.confidence ?? DEFAULT_THRESHOLDS.confidence,
  }
}

export function getGovernanceConfig(
  governance?: GovernanceConfig,
): Required<GovernanceConfig> {
  return {
    ttlDays: governance?.ttlDays ?? DEFAULT_GOVERNANCE.ttlDays,
    decayPerWeek: governance?.decayPerWeek ?? DEFAULT_GOVERNANCE.decayPerWeek,
    failureThreshold:
      governance?.failureThreshold ?? DEFAULT_GOVERNANCE.failureThreshold,
    minSuccessRate:
      governance?.minSuccessRate ?? DEFAULT_GOVERNANCE.minSuccessRate,
    recentWindowSize:
      governance?.recentWindowSize ?? DEFAULT_GOVERNANCE.recentWindowSize,
  }
}

export function getInputText(prompt: Prompt): string {
  if ('prompt' in prompt && typeof prompt.prompt === 'string') {
    return prompt.prompt
  }

  if ('messages' in prompt && Array.isArray(prompt.messages)) {
    for (let index = prompt.messages.length - 1; index >= 0; index -= 1) {
      const message = prompt.messages[index]
      if (message?.role !== 'user') {
        continue
      }

      return extractTextFromMessageContent(message.content)
    }
  }

  return ''
}

function extractTextFromMessageContent(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (!Array.isArray(content)) {
    return ''
  }

  return content
    .map(part => {
      if (typeof part === 'string') {
        return part
      }

      if (
        part != null &&
        typeof part === 'object' &&
        'type' in part &&
        (part as { type?: unknown }).type === 'text' &&
        'text' in part
      ) {
        return String((part as { text: unknown }).text)
      }

      return ''
    })
    .join(' ')
    .trim()
}

export function buildStopConditions<TOOLS>(
  maxSteps: number | undefined,
  stopWhen: unknown,
): unknown {
  if (maxSteps == null && stopWhen == null) {
    return stepCountIs(10)
  }

  if (maxSteps == null) {
    return stopWhen
  }

  const maxStepCondition = stepCountIs(maxSteps)

  if (stopWhen == null) {
    return maxStepCondition
  }

  return Array.isArray(stopWhen)
    ? [...stopWhen, maxStepCondition]
    : [stopWhen, maxStepCondition]
}

export function cosine(left: number[], right: number[]): number {
  if (left.length === 0 || right.length === 0 || left.length !== right.length) {
    return 0
  }

  return cosineSimilarity(left, right)
}

export function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9_-]+/i)
    .filter(token => token.length > 2 && !STOP_WORDS.has(token))
}

export function extractRecurringKeywords(inputs: string[], maxKeywords = 8): string[] {
  const counts = new Map<string, number>()

  for (const input of inputs) {
    const seen = new Set(tokenize(input))
    for (const token of seen) {
      counts.set(token, (counts.get(token) ?? 0) + 1)
    }
  }

  const minimumHits = Math.max(2, Math.ceil(inputs.length * 0.6))

  return [...counts.entries()]
    .filter(([, count]) => count >= minimumHits)
    .sort((left, right) => {
      if (right[1] !== left[1]) {
        return right[1] - left[1]
      }

      return left[0].localeCompare(right[0])
    })
    .slice(0, maxKeywords)
    .map(([token]) => token)
}

export function slugify(parts: string[]): string {
  const sanitized = parts
    .map(part => part.toLowerCase().replace(/[^a-z0-9]+/g, '-'))
    .map(part => part.replace(/^-+|-+$/g, ''))
    .filter(Boolean)

  return sanitized.join('-') || 'template'
}

export function shortHash(value: string): string {
  return createHash('sha1').update(value).digest('hex').slice(0, 8)
}

export function averageEmbedding(vectors: number[][]): number[] {
  if (vectors.length === 0) {
    return []
  }

  const size = vectors[0]?.length ?? 0
  const sums = new Array<number>(size).fill(0)

  for (const vector of vectors) {
    for (let index = 0; index < size; index += 1) {
      sums[index] = (sums[index] ?? 0) + (vector[index] ?? 0)
    }
  }

  return sums.map(sum => sum / vectors.length)
}

export function flattenObject(
  value: unknown,
  prefix = '',
  accumulator: Record<string, unknown> = {},
): Record<string, unknown> {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    if (prefix) {
      accumulator[prefix] = value
    }
    return accumulator
  }

  for (const [key, nestedValue] of Object.entries(value)) {
    const nextPrefix = prefix ? `${prefix}.${key}` : key
    flattenObject(nestedValue, nextPrefix, accumulator)
  }

  return accumulator
}

export function getByPath(value: unknown, path: string): unknown {
  if (!path) {
    return value
  }

  return path.split('.').reduce<unknown>((current, segment) => {
    if (current == null || typeof current !== 'object') {
      return undefined
    }

    if (Array.isArray(current)) {
      const index = Number(segment)
      return Number.isNaN(index) ? undefined : current[index]
    }

    return (current as Record<string, unknown>)[segment]
  }, value)
}

export function parseReference(
  value: unknown,
): { kind: 'input'; name: string } | { kind: 'node'; nodeId: string; path: string } | null {
  if (typeof value !== 'string' || !value.startsWith('{') || !value.endsWith('}')) {
    return null
  }

  const body = value.slice(1, -1)

  if (body.startsWith('input.')) {
    return { kind: 'input', name: body.slice('input.'.length) }
  }

  if (body.endsWith('.output')) {
    return {
      kind: 'node',
      nodeId: body.slice(0, -'.output'.length),
      path: '',
    }
  }

  const outputIndex = body.indexOf('.output.')
  if (outputIndex === -1) {
    return null
  }

  return {
    kind: 'node',
    nodeId: body.slice(0, outputIndex),
    path: body.slice(outputIndex + '.output.'.length),
  }
}

export function resolveGraphArgument(
  value: unknown,
  inputValues: Record<string, unknown>,
  nodeOutputs: Record<string, unknown>,
): unknown {
  const reference = parseReference(value)

  if (reference == null) {
    return value
  }

  if (reference.kind === 'input') {
    return inputValues[reference.name]
  }

  return getByPath(nodeOutputs[reference.nodeId], reference.path)
}

export function inferValueType(value: unknown): ArgValueType {
  switch (typeof value) {
    case 'number':
      return 'number'
    case 'boolean':
      return 'boolean'
    default:
      return 'string'
  }
}

export function inferRegex(values: unknown[]): string | undefined {
  if (values.length === 0) {
    return undefined
  }

  if (values.every(value => typeof value === 'number')) {
    return '-?\\d+(?:\\.\\d+)?'
  }

  if (values.every(value => typeof value === 'string')) {
    const strings = values as string[]

    if (strings.every(value => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value))) {
      return '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}'
    }

    if (strings.every(value => /^[A-Z]+-\d+$/.test(value))) {
      const stablePrefix = strings.every(value => /^([A-Z]+-)\d+$/.test(value))
        ? strings[0]?.match(/^([A-Z]+-)\d+$/)?.[1]
        : undefined

      if (stablePrefix != null) {
        return `${escapeRegex(stablePrefix)}\\d+`
      }

      const prefix = longestCommonPrefix(strings)
      if (prefix.length > 0) {
        return `${escapeRegex(prefix)}\\d+`
      }
    }

    if (strings.every(value => /^\d+$/.test(value))) {
      return '\\d+'
    }
  }

  return undefined
}

function longestCommonPrefix(values: string[]): string {
  if (values.length === 0) {
    return ''
  }

  let prefix = values[0] ?? ''
  for (const value of values.slice(1)) {
    while (!value.startsWith(prefix) && prefix.length > 0) {
      prefix = prefix.slice(0, -1)
    }
  }
  return prefix
}

export function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function sortEdges(edges: GraphEdge[]): GraphEdge[] {
  return [...edges].sort((left, right) => {
    if (left.from !== right.from) {
      return left.from.localeCompare(right.from)
    }

    if ((left.onFailure ?? false) !== (right.onFailure ?? false)) {
      return Number(left.onFailure ?? false) - Number(right.onFailure ?? false)
    }

    if (right.weight !== left.weight) {
      return right.weight - left.weight
    }

    return (left.to ?? '').localeCompare(right.to ?? '')
  })
}

export function sortNodes(nodes: Record<string, GraphNode>): Record<string, GraphNode> {
  return Object.keys(nodes)
    .sort()
    .reduce<Record<string, GraphNode>>((accumulator, key) => {
      accumulator[key] = nodes[key]!
      return accumulator
    }, {})
}

export function graphSignature(template: Pick<Template, 'graph' | 'args'>): string {
  return stableStringify({
    graph: {
      entryNode: template.graph.entryNode,
      nodes: sortNodes(template.graph.nodes),
      edges: sortEdges(template.graph.edges),
    },
    args: template.args,
  })
}

export function chooseLatestActiveTemplates(templates: Template[]): Template[] {
  const latestByFamily = new Map<string, Template>()

  for (const template of templates) {
    if (template.status !== 'active') {
      continue
    }

    const current = latestByFamily.get(template.templateFamilyId)
    if (current == null || template.version > current.version) {
      latestByFamily.set(template.templateFamilyId, template)
    }
  }

  return [...latestByFamily.values()].sort((left, right) =>
    left.id.localeCompare(right.id),
  )
}

export function scoreTemplateConfidence(input: {
  traceCount: number
  graphConsistency: number
  argumentResolution: number
  successRate: number
}): number {
  const volume = sigmoid((input.traceCount - 5) / 4)
  const score =
    volume * 0.35 +
    clamp(input.graphConsistency) * 0.3 +
    clamp(input.argumentResolution) * 0.2 +
    clamp(input.successRate) * 0.15

  return clamp(score, 0, 0.99)
}

export function countResolvedArguments(args: Record<string, { type: string }>): number {
  return Object.values(args).filter(arg => arg.type !== 'unresolved').length
}

export function createTraceStep(partial: Omit<TraceStep, 'id'>): TraceStep {
  return {
    id: createId('step'),
    ...partial,
  }
}
