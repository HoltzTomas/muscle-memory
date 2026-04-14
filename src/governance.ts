import type { Store } from './stores/interface'
import type {
  GovernanceConfig,
  MithrilEvents,
  Template,
  TemplateGovernanceState,
} from './types'
import {
  clamp,
  getGovernanceConfig,
  getEmbeddingModelIdentity,
} from './utils'

type EmitFn = <K extends keyof MithrilEvents>(
  eventName: K,
  payload: MithrilEvents[K],
) => void

export class GovernanceManager {
  private readonly config: Required<GovernanceConfig>

  constructor(
    private readonly store: Store,
    governance?: GovernanceConfig,
    private readonly emit?: EmitFn,
  ) {
    this.config = getGovernanceConfig(governance)
  }

  async ensureEmbeddingModelIdentity(model: { provider?: string; modelId?: string } | string) {
    const metadata = await this.store.getMetadata()
    const identity = getEmbeddingModelIdentity(model)

    if (
      metadata.embeddingModelId != null &&
      metadata.embeddingModelId !== identity
    ) {
      throw new Error(
        `Embedding model mismatch for store. Expected ${metadata.embeddingModelId} but received ${identity}.`,
      )
    }

    if (metadata.embeddingModelId == null) {
      await this.store.setMetadata({ embeddingModelId: identity })
    }
  }

  createInitialState(): TemplateGovernanceState {
    return {
      ttlDays: this.config.ttlDays,
      decayPerWeek: this.config.decayPerWeek,
      failureThreshold: this.config.failureThreshold,
      minSuccessRate: this.config.minSuccessRate,
      consecutiveFailures: 0,
      recentExecutions: [],
    }
  }

  getEffectiveConfidence(template: Template, now = new Date()): number {
    const lastTouchedAt = new Date(template.lastUsedAt ?? template.createdAt)
    const inactiveMs = Math.max(0, now.getTime() - lastTouchedAt.getTime())
    const inactiveWeeks = inactiveMs / (7 * 24 * 60 * 60 * 1000)

    return clamp(template.confidence - inactiveWeeks * this.config.decayPerWeek)
  }

  async syncTemplate(template: Template, now = new Date()): Promise<Template> {
    if (template.status !== 'active') {
      return template
    }

    if (!this.isExpired(template, now)) {
      return template
    }

    const degraded = {
      ...template,
      status: 'degraded' as const,
      governance: {
        ...template.governance,
        degradedReason: 'Template TTL expired',
      },
    }

    await this.store.updateTemplate(degraded.id, degraded)
    this.emit?.('template:degraded', {
      templateId: degraded.id,
      reason: 'Template TTL expired',
      successRate: degraded.successRate,
    })

    return degraded
  }

  async recordExecution(template: Template, success: boolean): Promise<Template> {
    const now = new Date().toISOString()
    const recentExecutions = [
      ...template.governance.recentExecutions,
      { success, timestamp: now },
    ].slice(-this.config.recentWindowSize)

    const successRate =
      recentExecutions.length === 0
        ? template.successRate
        : recentExecutions.filter(execution => execution.success).length /
          recentExecutions.length

    const consecutiveFailures = success
      ? 0
      : template.governance.consecutiveFailures + 1

    const degradedReason = this.getDegradedReason({
      ...template.governance,
      recentExecutions,
      consecutiveFailures,
    }, successRate)

    const updated: Template = {
      ...template,
      status: degradedReason ? 'degraded' : template.status,
      executionCount: template.executionCount + 1,
      lastUsedAt: now,
      successRate,
      governance: {
        ...template.governance,
        recentExecutions,
        consecutiveFailures,
        degradedReason,
      },
    }

    await this.store.updateTemplate(updated.id, updated)

    if (degradedReason != null && template.status !== 'degraded') {
      this.emit?.('template:degraded', {
        templateId: updated.id,
        reason: degradedReason,
        successRate: updated.successRate,
      })
    }

    return updated
  }

  async invalidate(templateId: string, reason: string): Promise<void> {
    const template = await this.store.getTemplate(templateId)
    if (template == null) {
      return
    }

    const degraded: Template = {
      ...template,
      status: 'degraded',
      governance: {
        ...template.governance,
        degradedReason: reason,
      },
    }

    await this.store.updateTemplate(templateId, degraded)
    this.emit?.('template:degraded', {
      templateId,
      reason,
      successRate: degraded.successRate,
    })
  }

  private isExpired(template: Template, now: Date): boolean {
    const lastTouchedAt = new Date(template.lastUsedAt ?? template.createdAt)
    const ageInDays =
      (now.getTime() - lastTouchedAt.getTime()) / (24 * 60 * 60 * 1000)

    return ageInDays >= this.config.ttlDays
  }

  private getDegradedReason(
    governance: TemplateGovernanceState,
    successRate: number,
  ): string | undefined {
    if (governance.consecutiveFailures >= this.config.failureThreshold) {
      return 'Template exceeded consecutive failure threshold'
    }

    if (
      governance.recentExecutions.length >= this.config.recentWindowSize &&
      successRate < this.config.minSuccessRate
    ) {
      return 'Template fell below minimum success rate'
    }

    return governance.degradedReason
  }
}
