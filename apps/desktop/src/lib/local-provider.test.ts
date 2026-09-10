import { describe, expect, it } from 'vitest'

import { isLocalProviderSlug } from './local-provider'

describe('isLocalProviderSlug', () => {
  it('treats both engines as the Local picker group', () => {
    expect(isLocalProviderSlug('llamacpp')).toBe(true)
    expect(isLocalProviderSlug('vllm')).toBe(true)
    expect(isLocalProviderSlug('llama.cpp')).toBe(true)
    expect(isLocalProviderSlug('nous')).toBe(false)
    expect(isLocalProviderSlug('')).toBe(false)
  })
})
