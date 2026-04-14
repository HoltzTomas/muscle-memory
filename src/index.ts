export { MithrilAgent } from './agent'
export { learn } from './learner'
export { inspect } from './inspect'
export { MemoryStore } from './stores/memory'
export type { Store } from './stores/interface'
export type * from './types'

import type { MithrilConfig } from './types'
import type { ToolSet as AISDKToolSet } from 'ai'
import { MithrilAgent } from './agent'

export function mithril<TOOLS extends AISDKToolSet>(
  config: MithrilConfig<TOOLS>,
): MithrilAgent<TOOLS> {
  return new MithrilAgent(config)
}
