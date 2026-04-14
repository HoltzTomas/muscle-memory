import type { Store } from './stores/interface'
import type { MatchResult, Template } from './types'
import { cosine, tokenize } from './utils'
import { GovernanceManager } from './governance'

export class Matcher {
  constructor(
    private readonly store: Store,
    private readonly governance: GovernanceManager,
    private readonly similarityThreshold: number,
    private readonly confidenceThreshold: number,
  ) {}

  async match(inputText: string, embedding: number[]): Promise<MatchResult | null> {
    const templates = await this.store.getActiveTemplates()
    const active = await this.getUsableTemplates(templates)

    if (active.length === 0) {
      return null
    }

    const keywords = new Set(tokenize(inputText))
    const keywordCandidates = active.filter(template =>
      template.keywords.some(keyword => keywords.has(keyword)),
    )

    return (
      this.findBest(keywordCandidates, embedding) ??
      this.findBest(active, embedding)
    )
  }

  private async getUsableTemplates(templates: Template[]): Promise<Template[]> {
    const usable: Template[] = []

    for (const template of templates) {
      const synced = await this.governance.syncTemplate(template)
      if (synced.status !== 'active') {
        continue
      }

      if (
        this.governance.getEffectiveConfidence(synced) <
        this.confidenceThreshold
      ) {
        continue
      }

      usable.push(synced)
    }

    return usable
  }

  private findBest(
    templates: Template[],
    embedding: number[],
  ): MatchResult | null {
    const scored = templates
      .map(template => ({
        template,
        similarity: cosine(template.centroid, embedding),
      }))
      .filter(result => result.similarity >= this.similarityThreshold)
      .sort((left, right) => {
        if (right.similarity !== left.similarity) {
          return right.similarity - left.similarity
        }

        return right.template.version - left.template.version
      })

    return scored[0] ?? null
  }
}
