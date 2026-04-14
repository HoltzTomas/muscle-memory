import { z } from 'zod'

import type { Template } from './types'
import type { AiRuntime } from './runtime'
import { inferValueType } from './utils'
import { mergeUsages } from './utils'

export class Extractor {
  constructor(
    private readonly runtime: AiRuntime,
    private readonly model: unknown,
  ) {}

  async extract(options: {
    inputText: string
    template: Template
    abortSignal?: AbortSignal | undefined
  }) {
    const extractableEntries = Object.entries(options.template.args)

    const unresolved = extractableEntries.filter(
      ([, definition]) => definition.type === 'unresolved',
    )

    if (unresolved.length > 0) {
      throw new Error(
        `Template ${options.template.id} still has unresolved arguments.`,
      )
    }

    const values: Record<string, unknown> = {}
    const missing: Array<{
      name: string
      definition: Extract<Template['args'][string], { type: 'extract_from_input' }>
    }> = []

    for (const [name, definition] of extractableEntries) {
      if (definition.type !== 'extract_from_input') {
        continue
      }

      const regexValue =
        definition.pattern == null
          ? undefined
          : extractWithRegex(options.inputText, definition.pattern, definition.valueType)

      if (regexValue !== undefined) {
        values[name] = regexValue
        continue
      }

      missing.push({ name, definition })
    }

    if (missing.length === 0) {
      return {
        values,
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      }
    }

    const schema = z.object(
      Object.fromEntries(
        missing.map(({ name, definition }) => [
          name,
          createFieldSchema(definition.valueType, definition.description),
        ]),
      ),
    )

    const result = await this.runtime.generateObject({
      model: this.model as never,
      schema,
      abortSignal: options.abortSignal,
      prompt: buildExtractionPrompt(options.inputText, missing),
    })

    return {
      values: {
        ...values,
        ...result.object,
      },
      usage: mergeUsages(result.usage),
      rawUsage: result.rawUsage,
    }
  }
}

function buildExtractionPrompt(
  inputText: string,
  missing: Array<{
    name: string
    definition: Extract<Template['args'][string], { type: 'extract_from_input' }>
  }>,
): string {
  const fieldDescriptions = missing
    .map(({ name, definition }) => {
      const description =
        definition.description ??
        `Extract ${name.replace(/_/g, ' ')} from the user request.`
      return `- ${name} (${definition.valueType}): ${description}`
    })
    .join('\n')

  return [
    'Extract the requested fields from the user message.',
    'Return values that exactly match the schema.',
    '',
    `User message: ${inputText}`,
    '',
    'Fields:',
    fieldDescriptions,
  ].join('\n')
}

function extractWithRegex(
  inputText: string,
  pattern: string,
  valueType: 'string' | 'number' | 'boolean',
): unknown {
  const match = inputText.match(new RegExp(pattern, 'i'))
  if (match == null) {
    return undefined
  }

  const rawValue = match[0]

  switch (valueType) {
    case 'number':
      return Number(rawValue)
    case 'boolean':
      return rawValue.toLowerCase() === 'true'
    default:
      return rawValue
  }
}

function createFieldSchema(
  valueType: 'string' | 'number' | 'boolean',
  description?: string,
) {
  switch (valueType ?? inferValueType('')) {
    case 'number':
      return z.number().describe(description ?? 'A numeric value.')
    case 'boolean':
      return z.boolean().describe(description ?? 'A true/false value.')
    default:
      return z.string().describe(description ?? 'A string value.')
  }
}
