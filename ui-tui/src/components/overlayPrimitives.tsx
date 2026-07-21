import type { Key } from '@hermes/ink'
import { Text, useInput } from '@hermes/ink'
import { type ReactNode, useState } from 'react'

import type { UsageModelData } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'

import { SelectableRow } from './selectableRow.js'

export interface MenuRowSpec {
  color?: string
  label: string
  run: () => void
}

/**
 * Mouse handlers a menu row can accept so a click highlights it and a click
 * on the highlight (or a double-click) activates it — the pointer twin of
 * ↑/↓ + Enter. Both are optional; a row given neither renders exactly as
 * before (keyboard-only), so adding these never changes existing callers.
 * `index0` is the row's 0-based position in the menu.
 */
export interface MenuRowMouse {
  index0?: number
  onActivate?: (index: number) => void
  onSelect?: (index: number) => void
}

export interface MenuControls {
  /** Current highlighted row (clamped to the row set). */
  sel: number
  /** Activate row `index` — the mouse/programmatic equivalent of Enter. */
  activate: (index: number) => void
  /** Move the highlight to row `index`. */
  select: (index: number) => void
}

/**
 * ↑/↓ + Enter + number-key selection over `rows`; Esc runs `onEscape`.
 * `onKey`, when given, runs first on every keypress — return `true` to mark
 * the key fully handled and skip the default escape/arrow/enter/number
 * handling for that keypress (e.g. a screen with a text-input sub-mode).
 *
 * Returns {@link MenuControls}: the highlighted index plus `select`/`activate`
 * so rows can also be driven by the mouse (see MenuRow/ActionRow mouse props).
 */
export function useMenu(
  rows: MenuRowSpec[],
  onEscape: () => void,
  onKey?: (ch: string, key: Key) => boolean
): MenuControls {
  const [sel, setSel] = useState(0)

  useInput((ch, key) => {
    if (onKey?.(ch, key)) {
      return
    }

    if (key.escape) {
      return onEscape()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)
    }

    if (key.downArrow && sel < rows.length - 1) {
      setSel(v => v + 1)
    }

    if (key.return) {
      return rows[sel]?.run()
    }

    const n = parseInt(ch, 10)

    if (n >= 1 && n <= rows.length) {
      return rows[n - 1]?.run()
    }
  })

  const clamp = (i: number) => Math.max(0, Math.min(rows.length - 1, i))

  return {
    activate: (i: number) => rows[clamp(i)]?.run(),
    sel: Math.min(sel, Math.max(0, rows.length - 1)),
    select: (i: number) => setSel(clamp(i))
  }
}

/** A numbered menu row with the ▸ cursor (mirrors ClarifyPrompt). */
export function MenuRow({
  active,
  index,
  index0,
  label,
  onActivate,
  onSelect,
  t
}: { active: boolean; index: number; label: string; t: Theme } & MenuRowMouse) {
  const inner = (
    <Text bold={active} color={active ? t.color.label : t.color.muted} inverse={active}>
      {active ? '▸ ' : '  '}
      {index}. {label}
    </Text>
  )

  if (onSelect && index0 !== undefined) {
    return (
      <SelectableRow index={index0} isActive={active} onActivate={onActivate} onSelect={onSelect}>
        {inner}
      </SelectableRow>
    )
  }

  return <Text>{inner}</Text>
}

/** Plain (non-numbered) action row with the ▸ cursor (confirm screens). */
export function ActionRow({
  active,
  color,
  index0,
  label,
  onActivate,
  onSelect,
  t
}: { active: boolean; label: string; color?: string; t: Theme } & MenuRowMouse) {
  const inner = (
    <>
      <Text color={active ? t.color.accent : t.color.muted}>{active ? '▸ ' : '  '}</Text>
      <Text bold={active} color={active ? (color ?? t.color.text) : t.color.muted}>
        {label}
      </Text>
    </>
  )

  if (onSelect && index0 !== undefined) {
    return (
      <SelectableRow index={index0} isActive={active} onActivate={onActivate} onSelect={onSelect}>
        {inner}
      </SelectableRow>
    )
  }

  return <Text>{inner}</Text>
}

export const BAR_CELLS = 10

/** ratio in [0,1] -> { bar: '█…░…', pct: 0-100 } using `cells` cells. */
export function barCells(ratio: number, cells: number = BAR_CELLS): { bar: string; pct: number } {
  const r = Math.max(0, Math.min(1, ratio))

  const filled = Math.round(r * cells)

  return { bar: '█'.repeat(filled) + '░'.repeat(cells - filled), pct: Math.round(r * 100) }
}

/**
 * Two-bar dollar usage view (decided with the user over a crammed three-segment
 * bar: at terminal widths a single fill glyph per full-resolution bar is the
 * only legible option). The plan bar is labeled with the plan name and shows
 * the allowance detail + % used; the top-up bar shows purchased dollars (no
 * denominator, renders full = balance, rolls over). Dollars only — never
 * "credits". Each row:
 *   `Plus    [██████░░░░]  $14.00 of $20.00 · 30% used`
 * Renders nothing for a free account (no bars to draw — caller shows upsell).
 */
export function UsageBars({ model, t }: { model: undefined | UsageModelData; t: Theme }) {
  if (!model || !model.available) {
    return null
  }

  const rows: ReactNode[] = []
  // Label the plan bar with the plan name (padded for column alignment with the
  // top-up row). Falls back to 'plan' when the name is absent.
  const planLabel = (model.plan_name || 'plan').padEnd(8).slice(0, 8)

  if (model.plan_bar) {
    const b = model.plan_bar
    const { bar } = barCells(b.fill_fraction)
    const pct = b.pct_used == null ? '' : ` · ${b.pct_used}% used`

    rows.push(
      <Text color={t.color.text} key="plan">
        {planLabel}
        <Text color={t.color.muted}>[</Text>
        <Text color={t.color.accent}>{bar}</Text>
        <Text color={t.color.muted}>]</Text>
        {`  ${b.remaining_display} left of ${b.total_display}${pct}`}
      </Text>
    )
  }

  if (model.topup_bar) {
    const b = model.topup_bar
    const { bar } = barCells(1)

    rows.push(
      <Text color={t.color.text} key="topup">
        {'top-up  '}
        <Text color={t.color.muted}>[</Text>
        <Text color={t.color.ok}>{bar}</Text>
        <Text color={t.color.muted}>]</Text>
        {`  ${b.remaining_display} · never expires`}
      </Text>
    )
  }

  if (rows.length === 0) {
    return null
  }

  return <>{rows}</>
}

/**
 * Plain-text version of the two-bar usage view, for text-only surfaces (the
 * /usage transcript panel). Returns one string per line: a plan bar, a top-up
 * bar, and a total-spendable summary, whichever apply. Dollars only.
 */
export function usageBarsText(model: undefined | UsageModelData): string[] {
  if (!model || !model.available) {
    return []
  }

  const lines: string[] = []
  const planLabel = (model.plan_name || 'plan').padEnd(8).slice(0, 8)

  if (model.plan_bar) {
    const b = model.plan_bar
    const { bar } = barCells(b.fill_fraction)
    const pct = b.pct_used == null ? '' : ` · ${b.pct_used}% used`

    lines.push(`${planLabel}[${bar}]  ${b.remaining_display} left of ${b.total_display}${pct}`)
  }

  if (model.topup_bar) {
    const b = model.topup_bar
    const { bar } = barCells(1)

    lines.push(`top-up  [${bar}]  ${b.remaining_display} · never expires`)
  }

  if (model.total_spendable_display && model.has_topup) {
    lines.push(`Total spendable: ${model.total_spendable_display}`)
  }

  return lines
}

export const footer = (extra: string, t: Theme) => <Text color={t.color.muted}>{extra}</Text>
