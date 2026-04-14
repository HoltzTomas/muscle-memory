import type {
  InspectGetTemplateGraphOptions,
  InspectInvalidateTemplateOptions,
  InspectListTemplatesOptions,
} from './types'
import { GovernanceManager } from './governance'

export const inspect = {
  async listTemplates({ store }: InspectListTemplatesOptions) {
    return store.getAllTemplates()
  },

  async getTemplateGraph({
    store,
    templateId,
  }: InspectGetTemplateGraphOptions) {
    const template = await store.getTemplate(templateId)
    if (template == null) {
      throw new Error(`Template ${templateId} was not found.`)
    }

    return template.graph
  },

  async invalidateTemplate({
    store,
    templateId,
    reason,
  }: InspectInvalidateTemplateOptions) {
    const governance = new GovernanceManager(store)
    await governance.invalidate(templateId, reason)
  },
}
