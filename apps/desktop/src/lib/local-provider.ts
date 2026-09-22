const LOCAL_SLUGS = new Set(['llamacpp', 'llama.cpp', 'llama-cpp', 'vllm'])
const LLAMA_SLUGS = new Set(['llamacpp', 'llama.cpp', 'llama-cpp'])

export const VLLM_GPU_ENDPOINT_NAME = 'vLLM GPU'
export const VLLM_CPU_ENDPOINT_NAME = 'vLLM CPU'

/** Chat picker / catalog group is Local regardless of llama.cpp vs vLLM. */
export function isLocalProviderSlug(slug: string | undefined | null): boolean {
  return LOCAL_SLUGS.has((slug ?? '').trim().toLowerCase())
}

function isLlamaCppSlug(slug: string | undefined | null): boolean {
  return LLAMA_SLUGS.has((slug ?? '').trim().toLowerCase())
}

/** llama.cpp stays "Local". A vLLM row keeps the device name the backend saved. */
export function providerGroupLabel(
  provider: { slug?: string | null; name?: string | null },
  localLabel: string
): string {
  if (isLlamaCppSlug(provider.slug)) return localLabel
  const name = (provider.name ?? '').trim()
  if (name) return name
  return isLocalProviderSlug(provider.slug) ? localLabel : (provider.slug ?? '').trim()
}

function loopbackPort(baseUrl: string): string | null {
  try {
    const url = new URL(baseUrl)
    const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase()
    const loopback =
      host === 'localhost' || host === '::1' || host === '0.0.0.0' || host === '127.0.0.1' || host.startsWith('127.')
    if (!loopback) return null
    if (url.port) return url.port
    if (url.protocol === 'https:') return '443'
    if (url.protocol === 'http:') return '80'
    return null
  } catch {
    return null
  }
}

/** Display name for a managed vLLM loopback URL. Other endpoints stay unlabeled. */
export function managedVllmEndpointName(baseUrl?: string | null): string | null {
  const port = loopbackPort((baseUrl ?? '').trim())
  if (port === '18436') return VLLM_CPU_ENDPOINT_NAME
  if (port === '18435') return VLLM_GPU_ENDPOINT_NAME
  return null
}

/** Aux task line: show the managed serve name when the stored URL is one of ours. */
export function auxTaskEndpointLabel(provider: string, baseUrl?: string | null): string {
  return managedVllmEndpointName(baseUrl) ?? provider
}

/** Picker header keeps the provider slug, except the two managed vLLM names. */
export function pickerHeaderProviderLabel(
  provider: { slug?: string | null; name?: string | null } | undefined,
  fallbackSlug: string,
  localLabel: string
): string {
  if (!provider) return fallbackSlug
  const label = providerGroupLabel(provider, localLabel)
  if (label === VLLM_GPU_ENDPOINT_NAME || label === VLLM_CPU_ENDPOINT_NAME) return label
  return fallbackSlug
}
