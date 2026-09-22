import { describe, expect, it } from 'vitest'

import {
  auxTaskEndpointLabel,
  isLocalProviderSlug,
  managedVllmEndpointName,
  pickerHeaderProviderLabel,
  providerGroupLabel
} from './local-provider'

describe('isLocalProviderSlug', () => {
  it('treats both engines as the Local picker group', () => {
    expect(isLocalProviderSlug('llamacpp')).toBe(true)
    expect(isLocalProviderSlug('vllm')).toBe(true)
    expect(isLocalProviderSlug('llama.cpp')).toBe(true)
    expect(isLocalProviderSlug('nous')).toBe(false)
    expect(isLocalProviderSlug('')).toBe(false)
  })
})

describe('managed vLLM endpoint names', () => {
  it('names the two loopback serves and leaves other URLs alone', () => {
    expect(managedVllmEndpointName('http://127.0.0.1:18435/v1')).toBe('vLLM GPU')
    expect(managedVllmEndpointName('http://localhost:18436/v1')).toBe('vLLM CPU')
    expect(managedVllmEndpointName('http://127.0.0.1:11434/v1')).toBeNull()
    expect(managedVllmEndpointName('http://127.0.0.1:18434/v1')).toBeNull()
    expect(managedVllmEndpointName('https://gpu.example:18435/v1')).toBeNull()
  })

  it('keeps llama.cpp under Local and shows each vLLM device name', () => {
    expect(providerGroupLabel({ slug: 'llamacpp', name: 'vLLM GPU' }, 'Local')).toBe('Local')
    expect(providerGroupLabel({ slug: 'vllm', name: 'vLLM GPU' }, 'Local')).toBe('vLLM GPU')
    expect(providerGroupLabel({ slug: 'custom', name: 'vLLM CPU' }, 'Local')).toBe('vLLM CPU')
    expect(pickerHeaderProviderLabel({ slug: 'custom', name: 'vLLM CPU' }, 'custom', 'Local')).toBe('vLLM CPU')
    expect(pickerHeaderProviderLabel({ slug: 'nous', name: 'Nous' }, 'nous', 'Local')).toBe('nous')
    expect(auxTaskEndpointLabel('custom', 'http://127.0.0.1:18436/v1')).toBe('vLLM CPU')
    expect(auxTaskEndpointLabel('custom', 'https://example.test/v1')).toBe('custom')
  })
})
