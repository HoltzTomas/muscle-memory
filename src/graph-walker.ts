import type { ToolSet, TypedToolCall, TypedToolResult } from 'ai'

import type { GraphWalkResult, Template, TraceStep } from './types'
import {
  createId,
  createTraceStep,
  getByPath,
  parseReference,
  resolveGraphArgument,
  serializeError,
} from './utils'

export class GraphWalker<TOOLS extends ToolSet = ToolSet> {
  constructor(private readonly tools: TOOLS) {}

  async execute(options: {
    template: Template
    inputValues: Record<string, unknown>
    messages?: unknown[] | undefined
    abortSignal?: AbortSignal | undefined
    experimentalContext?: unknown
  }): Promise<GraphWalkResult> {
    const outputsByNode: Record<string, unknown> = {}
    const steps: TraceStep[] = []
    let currentNodeId: string | null = options.template.graph.entryNode
    let iterations = 0

    while (currentNodeId != null) {
      iterations += 1
      if (iterations > Math.max(50, Object.keys(options.template.graph.nodes).length * 3)) {
        throw new Error('Graph walker exceeded safety iteration limit.')
      }

      const node = options.template.graph.nodes[currentNodeId]
      if (node == null) {
        throw new Error(`Graph node ${currentNodeId} was not found.`)
      }

      const tool = this.tools[node.tool]
      if (tool?.execute == null) {
        throw new Error(`Tool ${node.tool} is missing an execute function.`)
      }

      const resolvedArgs = Object.fromEntries(
        Object.entries(node.args).map(([key, value]) => [
          key,
          this.resolveArgument(value, options.inputValues, outputsByNode),
        ]),
      )

      const toolCallId = createId('toolcall')
      const startedAt = Date.now()

      try {
        const rawOutput = await tool.execute(resolvedArgs as never, {
          toolCallId,
          messages: (options.messages ?? []) as never,
          ...(options.abortSignal ? { abortSignal: options.abortSignal } : {}),
          ...(options.experimentalContext
            ? { experimental_context: options.experimentalContext }
            : {}),
        })

        const output = await resolveToolOutput(rawOutput)
        const latency = Date.now() - startedAt

        outputsByNode[currentNodeId] = output
        steps.push(
          createTraceStep({
            sequence: steps.length,
            stepNumber: steps.length,
            tool: node.tool,
            toolCallId,
            args: resolvedArgs,
            output,
            latency,
            success: true,
          }),
        )

        const nextEdge = chooseNextEdge(options.template, currentNodeId, output)
        if (nextEdge == null) {
          throw new Error(`No success edge matched for node ${currentNodeId}.`)
        }

        if (nextEdge.to == null) {
          return { steps, outputsByNode, terminalNodeId: currentNodeId }
        }

        currentNodeId = nextEdge.to
      } catch (error) {
        const latency = Date.now() - startedAt
        steps.push(
          createTraceStep({
            sequence: steps.length,
            stepNumber: steps.length,
            tool: node.tool,
            toolCallId,
            args: resolvedArgs,
            latency,
            success: false,
            error: serializeError(error),
          }),
        )

        const failureEdge = chooseFailureEdge(options.template, currentNodeId)
        if (failureEdge?.to == null) {
          throw error
        }

        if (failureEdge == null) {
          throw error
        }

        currentNodeId = failureEdge.to
      }
    }

    throw new Error('Graph walker finished without an explicit terminal edge.')
  }

  buildToolArtifacts(
    steps: TraceStep[],
  ): {
    toolCalls: Array<TypedToolCall<TOOLS>>
    toolResults: Array<TypedToolResult<TOOLS>>
  } {
    const toolCalls = steps.map(
      step =>
        ({
          type: 'tool-call',
          toolCallId: step.toolCallId,
          toolName: step.tool,
          input: step.args,
        }) as TypedToolCall<TOOLS>,
    )

    const toolResults = steps
      .filter(step => step.success)
      .map(
        step =>
          ({
            type: 'tool-result',
            toolCallId: step.toolCallId,
            toolName: step.tool,
            input: step.args,
            output: step.output,
          }) as TypedToolResult<TOOLS>,
      )

    return { toolCalls, toolResults }
  }

  private resolveArgument(
    value: unknown,
    inputValues: Record<string, unknown>,
    outputsByNode: Record<string, unknown>,
  ): unknown {
    const reference = parseReference(value)
    const resolved = resolveGraphArgument(value, inputValues, outputsByNode)

    if (reference != null && resolved === undefined) {
      throw new Error(`Unable to resolve reference ${String(value)}.`)
    }

    return resolved
  }
}

async function resolveToolOutput(value: unknown): Promise<unknown> {
  if (
    value != null &&
    typeof value === 'object' &&
    Symbol.asyncIterator in (value as Record<PropertyKey, unknown>)
  ) {
    let latest: unknown
    for await (const part of value as AsyncIterable<unknown>) {
      latest = part
    }
    return latest
  }

  return value
}

function chooseNextEdge(
  template: Template,
  nodeId: string,
  output: unknown,
) {
  return template.graph.edges
    .filter(edge => edge.from === nodeId && !(edge.onFailure ?? false))
    .filter(edge => matchesCondition(edge.condition, template, nodeId, output))
    .sort((left, right) => right.weight - left.weight)[0]
}

function chooseFailureEdge(template: Template, nodeId: string) {
  return template.graph.edges
    .filter(edge => edge.from === nodeId && (edge.onFailure ?? false))
    .sort((left, right) => right.weight - left.weight)[0]
}

function matchesCondition(
  condition: Template['graph']['edges'][number]['condition'],
  template: Template,
  nodeId: string,
  output: unknown,
): boolean {
  if (condition == null) {
    return true
  }

  const prefix = `${nodeId}.output.`
  const path = condition.field.startsWith(prefix)
    ? condition.field.slice(prefix.length)
    : condition.field

  const value = getByPath(output, path)

  if ('equals' in condition && condition.equals !== undefined) {
    return value === condition.equals
  }

  if ('notEquals' in condition && condition.notEquals !== undefined) {
    return value !== condition.notEquals
  }

  return true
}
