import type { Store } from './interface'
import type { StoreMetadata, Template, Trace } from '../types'
import { chooseLatestActiveTemplates, cosine } from '../utils'

export class MemoryStore implements Store {
  private readonly traces = new Map<string, Trace>()
  private readonly templates = new Map<string, Template>()
  private metadata: StoreMetadata = {}

  async saveTrace(trace: Trace): Promise<void> {
    this.traces.set(trace.id, structuredClone(trace))
  }

  async getUnprocessedTraces(): Promise<Trace[]> {
    return [...this.traces.values()]
      .filter(trace => !trace.processed)
      .sort((left, right) => {
        if (left.timestamp !== right.timestamp) {
          return left.timestamp.localeCompare(right.timestamp)
        }

        return left.id.localeCompare(right.id)
      })
      .map(trace => structuredClone(trace))
  }

  async markTracesProcessed(traceIds: string[]): Promise<void> {
    for (const traceId of traceIds) {
      const existing = this.traces.get(traceId)
      if (existing == null) {
        continue
      }

      this.traces.set(traceId, {
        ...existing,
        processed: true,
      })
    }
  }

  async saveTemplate(template: Template): Promise<void> {
    this.templates.set(template.id, structuredClone(template))
  }

  async getTemplate(id: string): Promise<Template | null> {
    const template = this.templates.get(id)
    return template == null ? null : structuredClone(template)
  }

  async getAllTemplates(): Promise<Template[]> {
    return [...this.templates.values()]
      .map(template => structuredClone(template))
      .sort((left, right) => left.id.localeCompare(right.id))
  }

  async getActiveTemplates(): Promise<Template[]> {
    return chooseLatestActiveTemplates([...this.templates.values()]).map(template =>
      structuredClone(template),
    )
  }

  async updateTemplate(id: string, updates: Partial<Template>): Promise<void> {
    const existing = this.templates.get(id)
    if (existing == null) {
      return
    }

    this.templates.set(id, structuredClone({ ...existing, ...updates }))
  }

  async deleteTemplate(id: string): Promise<void> {
    this.templates.delete(id)
  }

  async indexTemplateEmbedding(): Promise<void> {
    return
  }

  async findSimilarTemplates(
    embedding: number[],
    threshold: number,
  ): Promise<Array<{ templateId: string; similarity: number }>> {
    return chooseLatestActiveTemplates([...this.templates.values()])
      .map(template => ({
        templateId: template.id,
        similarity: cosine(template.centroid, embedding),
      }))
      .filter(result => result.similarity >= threshold)
      .sort((left, right) => right.similarity - left.similarity)
  }

  async getMetadata(): Promise<StoreMetadata> {
    return structuredClone(this.metadata)
  }

  async setMetadata(metadata: Partial<StoreMetadata>): Promise<void> {
    this.metadata = {
      ...this.metadata,
      ...metadata,
    }
  }
}
