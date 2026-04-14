import type { Store } from './stores/interface'
import type {
  ArgDefinition,
  GraphNode,
  LearnConfig,
  LearnResult,
  Template,
  Trace,
} from './types'
import {
  averageEmbedding,
  chooseLatestActiveTemplates,
  clamp,
  cosine,
  countResolvedArguments,
  deepEqual,
  extractRecurringKeywords,
  flattenObject,
  graphSignature,
  inferRegex,
  inferValueType,
  scoreTemplateConfidence,
  shortHash,
  slugify,
  stableStringify,
} from './utils'
import { GovernanceManager } from './governance'

const DEFAULT_MIN_TRACES = 5
const DEFAULT_CLUSTER_THRESHOLD = 0.82
const DEFAULT_CONFIDENCE_THRESHOLD = 0.9

export async function learn(config: LearnConfig): Promise<LearnResult> {
  const learner = new Learner(config.store)
  return learner.run(config)
}

export class Learner {
  constructor(private readonly store: Store) {}

  async run(config: LearnConfig): Promise<LearnResult> {
    const minTraces = config.minTraces ?? DEFAULT_MIN_TRACES
    const clusterThreshold = config.clusterThreshold ?? DEFAULT_CLUSTER_THRESHOLD
    const confidenceThreshold =
      config.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD

    const traces = await this.store.getUnprocessedTraces()
    const successful = traces
      .filter(trace => trace.success && trace.inputEmbedding.length > 0)
      .sort((left, right) => {
        if (left.timestamp !== right.timestamp) {
          return left.timestamp.localeCompare(right.timestamp)
        }

        return left.id.localeCompare(right.id)
      })

    const clusters = clusterTraces(successful, clusterThreshold)
    const existingTemplates = await this.store.getAllTemplates()
    const governance = new GovernanceManager(this.store)

    let templatesCreated = 0
    let templatesUpdated = 0

    for (const cluster of clusters) {
      if (cluster.length < minTraces) {
        continue
      }

      const graph = extractGraph(cluster)
      const { templateArgs, graphNodes, totalArguments, resolvedArguments } =
        abstractArguments(cluster, graph.nodes)

      const centroid = averageEmbedding(cluster.map(trace => trace.inputEmbedding))
      const keywords = extractRecurringKeywords(cluster.map(trace => trace.inputText))
      const graphConsistency = calculateGraphConsistency(graph.edges)
      const traceCount = cluster.length
      const priorTemplate = matchExistingTemplate(
        existingTemplates,
        centroid,
        graphSignature({ graph, args: templateArgs }),
        clusterThreshold,
      )

      const familyId =
        priorTemplate?.templateFamilyId ??
        createFamilyId(keywords, graphNodes, centroid, templateArgs)
      const version =
        priorTemplate == null
          ? 1
          : Math.max(
              ...existingTemplates
                .filter(template => template.templateFamilyId === familyId)
                .map(template => template.version),
              0,
            ) + 1

      const carriedTraceCount = priorTemplate?.traceCount ?? 0
      const carriedExecutions = priorTemplate?.executionCount ?? 0
      const successRate = priorTemplate?.successRate ?? 1
      const confidence = scoreTemplateConfidence({
        traceCount: traceCount + carriedTraceCount,
        graphConsistency,
        argumentResolution:
          totalArguments === 0 ? 1 : resolvedArguments / totalArguments,
        successRate,
      })

      const template: Template = {
        id: `${familyId}-v${version}`,
        templateFamilyId: familyId,
        version,
        status: confidence >= confidenceThreshold ? 'active' : 'learning',
        centroid,
        keywords,
        confidence,
        createdAt: new Date().toISOString(),
        lastUsedAt: priorTemplate?.lastUsedAt ?? null,
        executionCount: carriedExecutions,
        successRate,
        traceCount: carriedTraceCount + traceCount,
        args: templateArgs,
        graph: {
          entryNode: graph.entryNode,
          nodes: graphNodes,
          edges: graph.edges,
        },
        governance:
          priorTemplate?.governance ?? governance.createInitialState(),
      }

      await this.store.saveTemplate(template)
      await this.store.indexTemplateEmbedding(
        template.id,
        template.centroid,
        template.keywords,
      )

      existingTemplates.push(template)

      if (priorTemplate == null) {
        templatesCreated += 1
      } else {
        templatesUpdated += 1
      }
    }

    await this.store.markTracesProcessed(traces.map(trace => trace.id))

    return {
      clustersFound: clusters.length,
      templatesCreated,
      templatesUpdated,
      tracesProcessed: traces.length,
    }
  }
}

function clusterTraces(traces: Trace[], threshold: number): Trace[][] {
  const clusters: Trace[][] = []

  for (const trace of traces) {
    let bestClusterIndex = -1
    let bestSimilarity = -1

    for (let index = 0; index < clusters.length; index += 1) {
      const cluster = clusters[index]!
      const similarity = Math.max(
        ...cluster.map(candidate => cosine(candidate.inputEmbedding, trace.inputEmbedding)),
      )

      if (similarity >= threshold && similarity > bestSimilarity) {
        bestSimilarity = similarity
        bestClusterIndex = index
      }
    }

    if (bestClusterIndex === -1) {
      clusters.push([trace])
    } else {
      clusters[bestClusterIndex]!.push(trace)
    }
  }

  return clusters
}

function extractGraph(cluster: Trace[]) {
  const nodes: Record<string, GraphNode> = {}
  const sourceReachCount = new Map<string, number>()
  const edgeCounts = new Map<string, number>()
  const successEdgeExamples = new Map<
    string,
    Array<{ to: string | null; output: unknown }>
  >()

  for (const trace of cluster) {
    for (let index = 0; index < trace.steps.length; index += 1) {
      const step = trace.steps[index]!
      const nodeId = getNodeId(index, step.tool)
      nodes[nodeId] = {
        id: nodeId,
        tool: step.tool,
        args: {},
      }

      sourceReachCount.set(nodeId, (sourceReachCount.get(nodeId) ?? 0) + 1)

      const nextStep = trace.steps[index + 1]
      const to = nextStep == null ? null : getNodeId(index + 1, nextStep.tool)

      if (nextStep == null && !step.success) {
        continue
      }

      const onFailure = !step.success
      const edgeKey = stableStringify({ from: nodeId, to, onFailure })
      edgeCounts.set(edgeKey, (edgeCounts.get(edgeKey) ?? 0) + 1)

      if (!onFailure) {
        const examples = successEdgeExamples.get(nodeId) ?? []
        examples.push({ to, output: step.output })
        successEdgeExamples.set(nodeId, examples)
      }
    }
  }

  const edges = [...edgeCounts.entries()].map(([key, count]) => {
    const parsed = JSON.parse(key) as {
      from: string
      to: string | null
      onFailure: boolean
    }
    const reachCount = sourceReachCount.get(parsed.from) ?? count

    return {
      from: parsed.from,
      to: parsed.to,
      weight: clamp(count / reachCount),
      ...(parsed.onFailure ? { onFailure: true as const } : {}),
    }
  })

  inferConditions(edges, successEdgeExamples)

  const entryNode = cluster[0]?.steps[0]
    ? getNodeId(0, cluster[0].steps[0].tool)
    : ''

  return { entryNode, nodes, edges }
}

function inferConditions(
  edges: Array<{
    from: string
    to: string | null
    onFailure?: boolean
    weight: number
    condition?: { field: string; equals?: unknown }
  }>,
  examplesBySource: Map<string, Array<{ to: string | null; output: unknown }>>,
) {
  const edgesBySource = new Map<string, typeof edges>()

  for (const edge of edges) {
    if (edge.onFailure || edge.to == null) {
      continue
    }

    const group = edgesBySource.get(edge.from) ?? []
    group.push(edge)
    edgesBySource.set(edge.from, group)
  }

  for (const [source, sourceEdges] of edgesBySource.entries()) {
    if (new Set(sourceEdges.map(edge => edge.to)).size <= 1) {
      continue
    }

    const examples = examplesBySource.get(source) ?? []
    const groupedByTarget = new Map<string | null, Array<Record<string, unknown>>>()

    for (const example of examples) {
      const group = groupedByTarget.get(example.to) ?? []
      group.push(flattenObject(example.output))
      groupedByTarget.set(example.to, group)
    }

    const candidateFields = new Set<string>()
    for (const values of groupedByTarget.values()) {
      for (const example of values) {
        for (const key of Object.keys(example)) {
          candidateFields.add(key)
        }
      }
    }

    for (const field of candidateFields) {
      const targetValues = new Map<string | null, unknown>()
      let valid = true

      for (const [target, flattenedOutputs] of groupedByTarget.entries()) {
        const uniqueValues = [
          ...new Set(flattenedOutputs.map(output => stableStringify(output[field]))),
        ]

        if (uniqueValues.length !== 1) {
          valid = false
          break
        }

        targetValues.set(target, JSON.parse(uniqueValues[0]!))
      }

      if (!valid || new Set(targetValues.values()).size !== targetValues.size) {
        continue
      }

      for (const edge of sourceEdges) {
        edge.condition = {
          field: `${source}.output.${field}`,
          equals: targetValues.get(edge.to ?? null),
        }
      }

      break
    }
  }
}

function calculateGraphConsistency(
  edges: Array<{ from: string; weight: number }>,
): number {
  const grouped = new Map<string, number[]>()

  for (const edge of edges) {
    const weights = grouped.get(edge.from) ?? []
    weights.push(edge.weight)
    grouped.set(edge.from, weights)
  }

  if (grouped.size === 0) {
    return 1
  }

  const scores = [...grouped.values()].map(weights => Math.max(...weights))
  return scores.reduce((sum, score) => sum + score, 0) / scores.length
}

function abstractArguments(
  cluster: Trace[],
  nodes: Record<string, GraphNode>,
): {
  templateArgs: Record<string, ArgDefinition>
  graphNodes: Record<string, GraphNode>
  totalArguments: number
  resolvedArguments: number
} {
  const templateArgs: Record<string, ArgDefinition> = {}
  const graphNodes = Object.fromEntries(
    Object.entries(nodes).map(([key, node]) => [key, { ...node, args: {} }]),
  ) as Record<string, GraphNode>
  const usedArgNames = new Set<string>()
  let totalArguments = 0
  let resolvedArguments = 0

  const observationsByNode = new Map<
    string,
    Map<string, Array<{ value: unknown; inputText: string; trace: Trace; stepIndex: number }>>
  >()

  for (const trace of cluster) {
    for (let stepIndex = 0; stepIndex < trace.steps.length; stepIndex += 1) {
      const step = trace.steps[stepIndex]!
      const nodeId = getNodeId(stepIndex, step.tool)
      const byParam = observationsByNode.get(nodeId) ?? new Map()

      for (const [param, value] of Object.entries(step.args)) {
        const observations = byParam.get(param) ?? []
        observations.push({ value, inputText: trace.inputText, trace, stepIndex })
        byParam.set(param, observations)
      }

      observationsByNode.set(nodeId, byParam)
    }
  }

  for (const [nodeId, params] of observationsByNode.entries()) {
    const node = graphNodes[nodeId]!

    for (const [param, observations] of params.entries()) {
      totalArguments += 1

      const values = observations.map(observation => observation.value)
      if (values.every(value => deepEqual(value, values[0]))) {
        node.args[param] = values[0]
        resolvedArguments += 1
        continue
      }

      if (
        observations.every(observation =>
          appearsInInput(observation.value, observation.inputText),
        )
      ) {
        const argName = uniqueArgName(
          [node.tool, param],
          usedArgNames,
        )
        templateArgs[argName] = {
          type: 'extract_from_input',
          name: argName,
          pattern: inferRegex(values),
          description: `Value for ${param} when calling ${node.tool}.`,
          valueType: inferValueType(values[0]),
        }
        node.args[param] = `{input.${argName}}`
        resolvedArguments += 1
        continue
      }

      const chainedRef = inferChainedReference(observations)
      if (chainedRef != null) {
        node.args[param] = `{${chainedRef}}`
        resolvedArguments += 1
        continue
      }

      const argName = uniqueArgName(['unresolved', node.tool, param], usedArgNames)
      templateArgs[argName] = { type: 'unresolved' }
      node.args[param] = `{input.${argName}}`
    }
  }

  return { templateArgs, graphNodes, totalArguments, resolvedArguments }
}

function inferChainedReference(
  observations: Array<{
    value: unknown
    trace: Trace
    stepIndex: number
  }>,
): string | null {
  let candidateRefs: Set<string> | null = null

  for (const observation of observations) {
    const candidates = new Set<string>()

    for (let index = 0; index < observation.stepIndex; index += 1) {
      const prior = observation.trace.steps[index]!
      const priorNodeId = getNodeId(index, prior.tool)

      if (deepEqual(prior.output, observation.value)) {
        candidates.add(`${priorNodeId}.output`)
      }

      const flattened = flattenObject(prior.output)
      for (const [path, value] of Object.entries(flattened)) {
        if (deepEqual(value, observation.value)) {
          candidates.add(`${priorNodeId}.output.${path}`)
        }
      }
    }

    if (candidateRefs == null) {
      candidateRefs = candidates
      continue
    }

    candidateRefs = new Set(
      [...candidateRefs].filter(candidate => candidates.has(candidate)),
    )
  }

  return [...(candidateRefs ?? [])].sort()[0] ?? null
}

function appearsInInput(value: unknown, inputText: string): boolean {
  if (
    typeof value !== 'string' &&
    typeof value !== 'number' &&
    typeof value !== 'boolean'
  ) {
    return false
  }

  return inputText.toLowerCase().includes(String(value).toLowerCase())
}

function uniqueArgName(parts: string[], usedArgNames: Set<string>): string {
  const base = parts
    .join('_')
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'arg'

  let candidate = base
  let counter = 2

  while (usedArgNames.has(candidate)) {
    candidate = `${base}_${counter}`
    counter += 1
  }

  usedArgNames.add(candidate)
  return candidate
}

function createFamilyId(
  keywords: string[],
  nodes: Record<string, GraphNode>,
  centroid: number[],
  args: Record<string, ArgDefinition>,
): string {
  const toolNames = [...new Set(Object.values(nodes).map(node => node.tool))]
  const seed = slugify([...keywords.slice(0, 2), ...toolNames.slice(0, 1)])
  return `${seed}-${shortHash(
    stableStringify({
      tools: toolNames,
      centroid,
      args,
    }),
  )}`
}

function matchExistingTemplate(
  templates: Template[],
  centroid: number[],
  signature: string,
  threshold: number,
): Template | null {
  const latestTemplates = chooseLatestActiveTemplates(templates)

  const exact = latestTemplates.find(
    template => graphSignature(template) === signature,
  )

  if (exact != null) {
    return exact
  }

  return (
    latestTemplates
      .map(template => ({
        template,
        similarity: cosine(template.centroid, centroid),
      }))
      .filter(result => result.similarity >= threshold)
      .sort((left, right) => right.similarity - left.similarity)[0]?.template ??
    null
  )
}

function getNodeId(index: number, toolName: string): string {
  return `${index + 1}:${toolName}`
}
