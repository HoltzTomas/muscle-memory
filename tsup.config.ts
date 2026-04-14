import { defineConfig } from 'tsup'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/stores/memory.ts',
    'src/stores/redis.ts',
  ],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: false,
  clean: true,
  target: 'node18',
  splitting: false,
  treeshake: true,
})
