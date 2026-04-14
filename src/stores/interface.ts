import type { StoreMetadata, Template, Trace } from '../types'

export interface Store {
  saveTrace(trace: Trace): Promise<void>
  getUnprocessedTraces(): Promise<Trace[]>
  markTracesProcessed(traceIds: string[]): Promise<void>

  saveTemplate(template: Template): Promise<void>
  getTemplate(id: string): Promise<Template | null>
  getAllTemplates(): Promise<Template[]>
  getActiveTemplates(): Promise<Template[]>
  updateTemplate(id: string, updates: Partial<Template>): Promise<void>
  deleteTemplate(id: string): Promise<void>

  indexTemplateEmbedding(
    templateId: string,
    centroid: number[],
    keywords: string[],
  ): Promise<void>
  findSimilarTemplates(
    embedding: number[],
    threshold: number,
  ): Promise<Array<{ templateId: string; similarity: number }>>

  getMetadata(): Promise<StoreMetadata>
  setMetadata(metadata: Partial<StoreMetadata>): Promise<void>
}
