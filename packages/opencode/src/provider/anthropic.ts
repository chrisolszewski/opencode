import type { AssistantModelMessage, ModelMessage, ToolCallPart, ToolResultPart, UserModelMessage } from "ai"

type AssistantContent = Exclude<AssistantModelMessage["content"], string>

const SAFE_ID = /^[a-zA-Z0-9_-]+$/

export function normalizeAnthropicMessages(messages: ModelMessage[]): ModelMessage[] {
  const result: ModelMessage[] = []
  const idMap = new Map<string, string>()
  const reservedIds = new Set<string>()

  for (const msg of messages) {
    if ((msg.role === "assistant" || msg.role === "tool") && Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (hasToolCallId(part) && SAFE_ID.test(part.toolCallId)) {
          reservedIds.add(part.toolCallId)
        }
      }
    }
  }

  function sanitizeId(id: string): string {
    const existing = idMap.get(id)
    if (existing) return existing
    if (SAFE_ID.test(id)) {
      idMap.set(id, id)
      return id
    }
    let sanitized = `opencode_${Buffer.from(id).toString("base64url")}`
    while (reservedIds.has(sanitized)) {
      sanitized = `opencode_${Buffer.from(sanitized).toString("base64url")}`
    }
    idMap.set(id, sanitized)
    return sanitized
  }

  function collectWindow(startIndex: number, expectedIds: string[], expectedSet: Set<string>) {
    const collectedResults = new Map<string, ToolResultPart>()
    const syntheticMsgs: UserModelMessage[] = []

    let j = startIndex

    while (j < messages.length) {
      const next = messages[j]

      if (next.role === "tool") {
        if (!Array.isArray(next.content)) return

        for (const part of next.content) {
          if (!isToolResultPart(part)) return
          const sid = sanitizeId(part.toolCallId)
          if (!expectedSet.has(sid) || collectedResults.has(sid)) return
          collectedResults.set(sid, { ...part, toolCallId: sid })
        }

        j++
        continue
      }

      if (isSyntheticUserMessage(next)) {
        syntheticMsgs.push(next)
        j++
        continue
      }

      break
    }

    const hasAll = expectedIds.every((id) => collectedResults.has(id))
    if (!hasAll) return

    return {
      nextIndex: j,
      toolResults: expectedIds.map((id) => collectedResults.get(id)!),
      syntheticMsgs,
    }
  }

  let i = 0
  while (i < messages.length) {
    const msg = messages[i]

    if (msg.role === "tool") {
      if (!Array.isArray(msg.content)) {
        result.push(msg)
        i++
        continue
      }

      result.push({
        ...msg,
        content: msg.content.map((part) =>
          hasToolCallId(part) ? { ...part, toolCallId: sanitizeId(part.toolCallId) } : part,
        ),
      })
      i++
      continue
    }

    if (msg.role !== "assistant" || typeof msg.content === "string") {
      result.push(msg)
      i++
      continue
    }

    const isLast = i === messages.length - 1
    if (isLast && msg.content.some(isThinkingPart)) {
      result.push(msg)
      i++
      continue
    }

    const content = msg.content.map((part) =>
      hasToolCallId(part) ? { ...part, toolCallId: sanitizeId(part.toolCallId) } : part,
    ) satisfies AssistantContent

    const firstToolIdx = content.findIndex(isToolCallPart)
    if (firstToolIdx === -1) {
      result.push({ ...msg, content })
      i++
      continue
    }

    const beforeParts = content.slice(0, firstToolIdx)
    const toolCalls = content.slice(firstToolIdx).filter(isToolCallPart)
    const afterParts = content.slice(firstToolIdx).filter((p) => !isToolCallPart(p))

    const expectedIds = toolCalls.map((t) => t.toolCallId)
    const expectedSet = new Set(expectedIds)
    if (expectedSet.size !== expectedIds.length) {
      result.push({ ...msg, content })
      i++
      continue
    }

    const window = collectWindow(i + 1, expectedIds, expectedSet)
    if (!window) {
      result.push({ ...msg, content })
      i++
      continue
    }

    const mainContent = [...beforeParts, ...toolCalls]
    result.push({ ...msg, content: mainContent })
    result.push({ role: "tool", content: window.toolResults })
    result.push(...window.syntheticMsgs)

    if (afterParts.length > 0) {
      result.push({ role: "assistant", content: afterParts })
    }

    i = window.nextIndex
  }

  return result
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v)
}

function hasToolCallId(part: unknown): part is { toolCallId: string } {
  return isRecord(part) && typeof part.toolCallId === "string" && part.toolCallId.length > 0
}

function isToolCallPart(part: unknown): part is ToolCallPart {
  return isRecord(part) && part.type === "tool-call" && hasToolCallId(part)
}

function isToolResultPart(part: unknown): part is ToolResultPart {
  return isRecord(part) && part.type === "tool-result" && hasToolCallId(part)
}

function isThinkingPart(part: unknown): boolean {
  return isRecord(part) && (part.type === "thinking" || part.type === "redacted_thinking")
}

function isSyntheticUserMessage(msg: ModelMessage): msg is UserModelMessage {
  if (msg.role !== "user" || !Array.isArray(msg.content)) return false

  const hasFilePart = msg.content.some((part) => isRecord(part) && part.type === "file")
  if (!hasFilePart) return false

  return msg.content.some((part) => {
    if (!isRecord(part)) return false
    const meta = (part as any).providerMetadata?.opencode
    return meta?.synthetic === true
  })
}
