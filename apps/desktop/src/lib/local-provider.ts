const LOCAL_SLUGS = new Set(['llamacpp', 'llama.cpp', 'llama-cpp', 'vllm'])

/** Chat picker / catalog group is Local regardless of llama.cpp vs vLLM. */
export function isLocalProviderSlug(slug: string | undefined | null): boolean {
  return LOCAL_SLUGS.has((slug ?? '').trim().toLowerCase())
}
