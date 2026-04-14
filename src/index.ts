export { MuscleMemoryAgent } from './agent'
export { learn } from './learner'
export { inspect } from './inspect'
export { MemoryStore } from './stores/memory'
export type { Store } from './stores/interface'
export type * from './types'

import type { MuscleMemoryConfig } from './types'
import type { ToolSet as AISDKToolSet } from 'ai'
import { MuscleMemoryAgent } from './agent'

export function muscleMemory<TOOLS extends AISDKToolSet>(
  config: MuscleMemoryConfig<TOOLS>,
): MuscleMemoryAgent<TOOLS> {
  return new MuscleMemoryAgent(config)
}
