// Панель управления: HUD, камера, воспроизведение, раскраска, модель рельсов,
// отображение, легенда. Полупрозрачная, тёмная, со своей прокруткой и сворачиванием
// по Tab (состояние прячет сам App, отсюда приходит onlyHide).

import type { ReactNode } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { CheckRow, GroupTitle, SliderRow, SwitchRow } from '@/panel/controls'
import { Hud } from '@/panel/Hud'
import { Legend } from '@/panel/Legend'
import type { ViewerController } from '@/state/useViewer'
import { PRESETS } from '@/viewer/constants'

/** Пояснения к техническим именам режимов раскраски. */
const MODE_LABELS: Record<string, string> = {
  height: 'высота',
  intensity: 'интенсивность',
  range: 'дальность',
  ring: 'луч',
  zones: 'зоны',
}

const SPEED_MIN = 1
const SPEED_MAX = 30

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-1 pt-2">
      <GroupTitle>{title}</GroupTitle>
      {children}
    </div>
  )
}

export interface PanelProps {
  viewer: ViewerController
  onHide: () => void
}

export function Panel({ viewer, onHide }: PanelProps) {
  const {
    meta,
    idx,
    playing,
    rateHz,
    realtime,
    recordingPeriodMs,
    mode,
    pointSize,
    maxDist,
    grid,
    axes,
    additive,
    rails,
    railRange,
    stats,
    loading,
    preset,
  } = viewer

  if (!meta) return null

  return (
    // Высота панели задана явно (top-3 + bottom-3): иначе ScrollArea со своим
    // Viewport=100% не получает определённой высоты и не прокручивается.
    <div className="pointer-events-auto absolute top-3 bottom-3 left-3 flex w-[294px] flex-col overflow-hidden rounded-[10px] border border-white/15 bg-[#0c0f18]/70 shadow-[0_8px_28px_rgba(0,0,0,.45)] backdrop-blur-md">
      <div className="flex items-center justify-between gap-2 px-3 pt-3">
        <Badge variant="outline" className="max-w-[190px] overflow-hidden text-[11px]">
          {meta.bag}
        </Badge>
        <div className="flex items-center gap-1">
          {loading && <span className="text-[11px] text-muted-foreground">загрузка…</span>}
          <Button variant="ghost" size="xs" onClick={onHide} title="Свернуть панель (Tab)">
            Свернуть · Tab
          </Button>
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="px-3 pb-3">
          <Hud meta={meta} idx={idx} stats={stats} />

          <Group title="Камера">
            <div className="flex flex-wrap gap-1">
              {PRESETS.map(([key, label]) => (
                <Button
                  key={key}
                  size="xs"
                  variant={preset === key ? 'secondary' : 'outline'}
                  className="min-w-[62px] flex-1"
                  onClick={() => viewer.applyPreset(key)}
                >
                  {label}
                </Button>
              ))}
            </div>
          </Group>

          <Group title="Воспроизведение">
            <div className="flex flex-wrap gap-1">
              <Button size="xs" variant="secondary" className="min-w-[62px] flex-1" onClick={viewer.togglePlay}>
                {playing ? 'Пауза' : 'Играть'}
              </Button>
              {[-10, -1, 1, 10].map((delta) => (
                <Button
                  key={delta}
                  size="xs"
                  variant="outline"
                  className="min-w-[46px] flex-1"
                  onClick={() => viewer.stepBy(delta)}
                >
                  {delta > 0 ? `+${delta}` : `−${Math.abs(delta)}`}
                </Button>
              ))}
            </div>
            <SliderRow
              label="Кадр"
              value={idx}
              min={0}
              max={Math.max(meta.frames - 1, 0)}
              step={1}
              onChange={viewer.seek}
              format={(v) => String(v + 1)}
            />
            <SliderRow
              label="Скорость, Гц"
              value={rateHz}
              min={SPEED_MIN}
              max={SPEED_MAX}
              step={1}
              onChange={viewer.setRateHz}
              disabled={realtime}
            />
            <SwitchRow
              label="реальное время (по таймстемпам)"
              checked={realtime}
              onChange={viewer.setRealtime}
            />
            {realtime && (
              <div className="text-[11px] text-muted-foreground">
                кадры идут по таймстемпам записи: {recordingPeriodMs.toFixed(1)} мс ≈{' '}
                {(1000 / Math.max(recordingPeriodMs, 1)).toFixed(1)} Гц
              </div>
            )}
          </Group>

          <Group title="Раскраска">
            <Select value={mode} onValueChange={(next) => next && viewer.setMode(String(next))}>
              <SelectTrigger className="w-full" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {meta.modes.map((m) => (
                  <SelectItem key={m} value={m}>
                    {MODE_LABELS[m] ? `${m} — ${MODE_LABELS[m]}` : m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Group>

          <Group title="Рельсы (модель, 3 шт)">
            <SliderRow
              label="Ось X, м"
              value={rails.axis}
              min={railRange.axisMin}
              max={railRange.axisMax}
              step={0.02}
              onChange={(axis) => viewer.setRail({ axis })}
              format={(v) => v.toFixed(2)}
            />
            <SliderRow
              label="Колея, мм"
              value={rails.gaugeMm}
              min={railRange.gaugeMin}
              max={railRange.gaugeMax}
              step={5}
              onChange={(gaugeMm) => viewer.setRail({ gaugeMm })}
            />
            <SwitchRow
              label="контактный рельс сбоку"
              checked={rails.contact}
              onChange={(contact) => viewer.setRail({ contact })}
            />
            <SliderRow
              label="Смещение, м"
              value={rails.offset}
              min={railRange.offsetMin}
              max={railRange.offsetMax}
              step={0.02}
              onChange={(offset) => viewer.setRail({ offset })}
              format={(v) => v.toFixed(2)}
              disabled={!rails.contact}
            />
            <SwitchRow
              label="контактный со стороны +X"
              checked={rails.side === 1}
              onChange={(on) => viewer.setRail({ side: on ? 1 : -1 })}
              disabled={!rails.contact}
            />
            <div className="text-[11px] leading-snug text-muted-foreground">
              Ходовые рельсы: ось ± колея/2. Контактный: ось + смещение.
              <br />
              Пол пересчитывается по полосе вокруг оси.
            </div>
          </Group>

          <Group title="Отображение">
            <SliderRow
              label="Размер точки"
              value={pointSize}
              min={1}
              max={12}
              step={0.5}
              onChange={viewer.setPointSize}
            />
            <SliderRow
              label="Дальность, м"
              value={maxDist}
              min={5}
              max={200}
              step={5}
              onChange={viewer.setMaxDist}
            />
            <CheckRow label="сетка" checked={grid} onChange={viewer.setGrid} />
            <CheckRow label="оси" checked={axes} onChange={viewer.setAxes} />
            <CheckRow label="плотная заливка" checked={additive} onChange={viewer.setAdditive} />
          </Group>

          <Group title="Легенда">
            <Legend meta={meta} />
          </Group>

          <Separator className="mt-3" />
          <div className="pt-2 text-[11px] text-muted-foreground">
            ЛКМ — поворот, ПКМ/средняя — сдвиг, колесо — зум
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}