import { beforeEach, expect, test } from 'vitest'

import { $notifications, clearNotifications, isDiskFullErrorMessage, notifyError } from './notifications'

beforeEach(() => {
  clearNotifications()
})

function lastMessage(): string {
  return $notifications.get()[0]?.message ?? ''
}

// Regression for #39365: a gateway auth 401 (bad API_SERVER_KEY) must not be
// summarized as a provider (OpenAI/OpenRouter) API key problem.
test('gateway_auth_failed error is summarized as gateway auth, not provider key', () => {
  notifyError(
    new Error(
      '401 {"error": {"message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error", "code": "gateway_auth_failed"}}'
    ),
    'Request failed'
  )

  expect(lastMessage()).toContain('API_SERVER_KEY')
  expect(lastMessage()).not.toMatch(/OpenAI/i)
})

test('provider invalid_api_key error still maps to the OpenAI summary', () => {
  notifyError(
    new Error('401 {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}'),
    'Request failed'
  )

  expect(lastMessage()).toMatch(/OpenAI rejected the API key/i)
})

test('disk-full / ENOSPC errors toast a free-space message', () => {
  expect(isDiskFullErrorMessage('OSError: [Errno 28] No space left on device')).toBe(true)
  expect(isDiskFullErrorMessage('sqlite3.OperationalError: database or disk is full')).toBe(true)
  expect(isDiskFullErrorMessage('disk full: session storage could not be written — free some disk space')).toBe(true)
  expect(isDiskFullErrorMessage('This is often a full disk — free some space')).toBe(true)
  expect(isDiskFullErrorMessage('session storage could not be written: permission denied')).toBe(false)
  expect(isDiskFullErrorMessage('network timeout')).toBe(false)

  notifyError(new Error('OSError: [Errno 28] No space left on device: state.db'), 'Prompt failed')

  expect(lastMessage()).toMatch(/Disk full/i)
  expect(lastMessage()).toMatch(/free some space/i)
})

test('session storage write failure is treated as disk-full class', () => {
  notifyError(
    new Error('disk full: session storage could not be written — free some disk space and try again'),
    'Prompt failed'
  )

  expect(lastMessage()).toMatch(/Disk full/i)
})

test('code-skew 503 unwraps to a restart-required summary, not raw IPC JSON', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 503: {"detail":"Restart required: This process is running code from 08b4875f4a but the checkout on disk is now 48d2528066."}'
    ),
    'Could not load models'
  )

  expect(lastMessage()).toMatch(/running old code after an update/i)
  expect(lastMessage()).not.toMatch(/hermes:api/)
  expect(lastMessage()).not.toMatch(/systemctl/)
})

test('vLLM 400 JSON detail is the toast, not a raw Bad Request', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 400: {"detail":"This Hugging Face repo is gated — sign in at huggingface.co and request access. Hermes will not download it unsigned."}'
    ),
    'Could not set this as the local model'
  )

  expect(lastMessage()).toMatch(/gated/i)
  expect(lastMessage()).not.toMatch(/Bad Request/i)
  expect(lastMessage()).not.toMatch(/hermes:api/)
})

test('vLLM too-big needs-AWQ 400 is the toast, not the set-failed title', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 400: {"detail":"This full-precision checkpoint is too big for this GPU at the 64k tool-loop floor — Download an AWQ or FP8 instruct model"}'
    ),
    'Could not set this as the local model'
  )

  expect(lastMessage()).toMatch(/AWQ/i)
  expect(lastMessage()).toMatch(/too big/i)
  expect(lastMessage()).not.toBe('Could not set this as the local model')
  expect(lastMessage()).not.toMatch(/Bad Request/i)
})

test('empty-body 400 Bad Request uses the fallback, not statusText', () => {
  notifyError(new Error('400: Bad Request'), 'Local model setup failed')

  expect(lastMessage()).toBe('Local model setup failed')
  expect(lastMessage()).not.toMatch(/400/)
})

test('502 wrapping urllib HTTP Error 400 uses the fallback, not nested Bad Request', () => {
  notifyError(
    new Error(
      'Error invoking remote method \'hermes:api\': Error: 502: {"detail":"HTTP Error 400: Bad Request"}'
    ),
    'Could not set this as the local model'
  )

  expect(lastMessage()).toBe('Could not set this as the local model')
  expect(lastMessage()).not.toMatch(/Bad Request/i)
  expect(lastMessage()).not.toMatch(/hermes:api/)
  expect(lastMessage()).not.toMatch(/502/)
})
