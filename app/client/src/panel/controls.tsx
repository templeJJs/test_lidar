// Мелкие элементы панели: заголовок группы, строка с ползунком, строки-переключатели.
// Собраны на компонентах shadcn, чтобы панель не разъезжалась по стилю.

import { useId, type ReactNode } from 'react'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'

export function GroupTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('text-[11px] tracking-[0.06em] text-muted-foreground uppercase', className)}>
      {children}
    </div>
  )
}

function firstNumber(value: number | readonly number[]): number {
  return typeof value === 'number' ? value : value[0]
}

export interface SliderRowProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  format?: (value: number) => string
  disabled?: boolean
}

export function SliderRow({ label, value, min, max, step, onChange, format, disabled }: SliderRowProps) {
  const id = useId()
  return (
    <div className="flex items-center gap-2 py-0.5">
      <Label htmlFor={id} className="w-24 shrink-0 text-xs font-normal text-muted-foreground">
        {label}
      </Label>
      <Slider
        id={id}
        className="min-w-0 flex-1"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        value={[value]}
        onValueChange={(next) => {
          const num = firstNumber(next)
          if (Number.isFinite(num)) onChange(num)
        }}
      />
      <b className="w-10 shrink-0 text-right text-xs font-semibold tabular-nums">
        {format ? format(value) : String(value)}
      </b>
    </div>
  )
}

export interface CheckRowProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}

export function CheckRow({ label, checked, onChange }: CheckRowProps) {
  const id = useId()
  return (
    <div className="flex items-center gap-2 py-0.5">
      <Checkbox id={id} checked={checked} onCheckedChange={(next) => onChange(next)} />
      <Label htmlFor={id} className="text-xs font-normal text-muted-foreground">
        {label}
      </Label>
    </div>
  )
}

export function SwitchRow({ label, checked, onChange, disabled }: CheckRowProps & { disabled?: boolean }) {
  const id = useId()
  return (
    <div className="flex items-center gap-2 py-0.5">
      <Switch id={id} checked={checked} onCheckedChange={(next) => onChange(next)} disabled={disabled} />
      <Label htmlFor={id} className="text-xs font-normal text-muted-foreground">
        {label}
      </Label>
    </div>
  )
}