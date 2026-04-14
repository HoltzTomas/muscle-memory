import { createClient, type RedisClientType } from 'redis'

import type { Store } from './interface'
import type { StoreMetadata, Template, Trace } from '../types'
import { chooseLatestActiveTemplates, cosine } from '../utils'

export interface RedisStoreOptions {
  url: string
  keyPrefix?: string
}

export class RedisStore implements Store {
  private readonly client: RedisClientType
  private readonly keyPrefix: string
  private connectPromise?: Promise<void>

  constructor(options: RedisStoreOptions) {
    this.client = createClient({ url: options.url })
    this.keyPrefix = options.keyPrefix ?? 'mithril'
  }

  async saveTrace(trace: Trace): Promise<void> {
    await this.ensureConnected()
    await this.client.set(this.traceKey(trace.id), JSON.stringify(trace))
    await this.client.sAdd(this.tracesKey(), trace.id)
    if (!trace.processed) {
      await this.client.sAdd(this.unprocessedTracesKey(), trace.id)
    }
  }

  async getUnprocessedTraces(): Promise<Trace[]> {
    await this.ensureConnected()
    const ids = await this.client.sMembers(this.unprocessedTracesKey())
    return this.loadMany<Trace>(ids, this.traceKey.bind(this))
  }

  async markTracesProcessed(traceIds: string[]): Promise<void> {
    await this.ensureConnected()
    if (traceIds.length === 0) {
      return
    }

    for (const traceId of traceIds) {
      const trace = await this.getByKey<Trace>(this.traceKey(traceId))
      if (trace == null) {
        continue
      }

      trace.processed = true
      await this.client.set(this.traceKey(traceId), JSON.stringify(trace))
    }

    await this.client.sRem(this.unprocessedTracesKey(), traceIds)
  }

  async saveTemplate(template: Template): Promise<void> {
    await this.ensureConnected()
    await this.client.set(this.templateKey(template.id), JSON.stringify(template))
    await this.client.sAdd(this.templatesKey(), template.id)

    if (template.status === 'active') {
      await this.client.hSet(
        this.activeFamiliesKey(),
        template.templateFamilyId,
        template.id,
      )
    }
  }

  async getTemplate(id: string): Promise<Template | null> {
    await this.ensureConnected()
    return this.getByKey<Template>(this.templateKey(id))
  }

  async getAllTemplates(): Promise<Template[]> {
    await this.ensureConnected()
    const ids = await this.client.sMembers(this.templatesKey())
    return this.loadMany<Template>(ids, this.templateKey.bind(this))
  }

  async getActiveTemplates(): Promise<Template[]> {
    await this.ensureConnected()
    const familyMap = await this.client.hGetAll(this.activeFamiliesKey())
    const ids = Object.values(familyMap)
    const templates = await this.loadMany<Template>(ids, this.templateKey.bind(this))
    return chooseLatestActiveTemplates(templates)
  }

  async updateTemplate(id: string, updates: Partial<Template>): Promise<void> {
    await this.ensureConnected()
    const existing = await this.getTemplate(id)
    if (existing == null) {
      return
    }

    await this.saveTemplate({ ...existing, ...updates })
  }

  async deleteTemplate(id: string): Promise<void> {
    await this.ensureConnected()
    const template = await this.getTemplate(id)
    await this.client.del(this.templateKey(id))
    await this.client.sRem(this.templatesKey(), id)

    if (template != null) {
      await this.client.hDel(this.activeFamiliesKey(), template.templateFamilyId)
    }
  }

  async indexTemplateEmbedding(
    templateId: string,
    _centroid: number[],
    keywords: string[],
  ): Promise<void> {
    await this.ensureConnected()
    for (const keyword of keywords) {
      await this.client.sAdd(this.keywordKey(keyword), templateId)
    }
  }

  async findSimilarTemplates(
    embedding: number[],
    threshold: number,
  ): Promise<Array<{ templateId: string; similarity: number }>> {
    const templates = await this.getActiveTemplates()
    return templates
      .map(template => ({
        templateId: template.id,
        similarity: cosine(template.centroid, embedding),
      }))
      .filter(result => result.similarity >= threshold)
      .sort((left, right) => right.similarity - left.similarity)
  }

  async getMetadata(): Promise<StoreMetadata> {
    await this.ensureConnected()
    return (await this.getByKey<StoreMetadata>(this.metadataKey())) ?? {}
  }

  async setMetadata(metadata: Partial<StoreMetadata>): Promise<void> {
    await this.ensureConnected()
    const existing = await this.getMetadata()
    await this.client.set(this.metadataKey(), JSON.stringify({ ...existing, ...metadata }))
  }

  private async ensureConnected() {
    if (this.client.isOpen) {
      return
    }

    this.connectPromise ??= this.client.connect().then(() => undefined)
    await this.connectPromise
  }

  private async getByKey<T>(key: string): Promise<T | null> {
    const payload = await this.client.get(key)
    return payload == null ? null : (JSON.parse(payload) as T)
  }

  private async loadMany<T>(
    ids: string[],
    keyBuilder: (id: string) => string,
  ): Promise<T[]> {
    if (ids.length === 0) {
      return []
    }

    const payloads = await this.client.mGet(ids.map(id => keyBuilder(id)))
    return payloads
      .filter((payload): payload is string => payload != null)
      .map(payload => JSON.parse(payload) as T)
  }

  private traceKey(id: string) {
    return `${this.keyPrefix}:trace:${id}`
  }

  private templateKey(id: string) {
    return `${this.keyPrefix}:template:${id}`
  }

  private tracesKey() {
    return `${this.keyPrefix}:traces`
  }

  private unprocessedTracesKey() {
    return `${this.keyPrefix}:traces:unprocessed`
  }

  private templatesKey() {
    return `${this.keyPrefix}:templates`
  }

  private activeFamiliesKey() {
    return `${this.keyPrefix}:templates:active-families`
  }

  private keywordKey(keyword: string) {
    return `${this.keyPrefix}:keyword:${keyword}`
  }

  private metadataKey() {
    return `${this.keyPrefix}:metadata`
  }
}
