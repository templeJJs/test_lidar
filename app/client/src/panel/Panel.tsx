// Панель управления: HUD, камера, воспроизведение, раскраска, линия хода,
// туннель безопасности, модель рельсов, отображение, легенда. Полупрозрачная,
// тёмная, со своей прокруткой и сворачиванием по Tab (состояние прячет сам App,
// отсюда приходит onlyHide).

import type { ReactNode } from 'react'
import { PATH_VARIANT_LABELS, type PathVariantName } from '@/api/pathTunnel'
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
import { CheckRow, GroupTitle, Note, SliderRow, SwitchRow } from '@/panel/controls'
import { Hud } from '@/panel/Hud'
import { LabelPanel } from '@/panel/LabelPanel'
import { ObjectsPanel } from '@/panel/ObjectsPanel'
import { Legend } from '@/panel/Legend'
import type { PathSource, ViewerController } from '@/state/useViewer'
import type { LabelingController } from '@/state/useLabeling'
import {
  PATH_THICK_MAX,
  PATH_THICK_MIN,
  PRESETS,
  TUNNEL_HALF_MAX,
  TUNNEL_HALF_MIN,
  TUNNEL_HIGH_MAX,
  TUNNEL_HIGH_MIN,
  TUNNEL_LOW_MAX,
  TUNNEL_LOW_MIN,
} from '@/viewer/constants'

/** Пояснения к техническим именам режимов раскраски. */
const MODE_LABELS: Record<string, string> = {
  height: 'высота',
  intensity: 'интенсивность',
  range: 'дальность',
  ring: 'луч',
  zones: 'зоны',
}

/** Источники линии хода для селекта: три варианта и режим сравнения. */
const PATH_SOURCE_ITEMS: ReadonlyArray<readonly [PathSource, string]> = [
  ['corridor', PATH_VARIANT_LABELS.corridor],
  ['rails', PATH_VARIANT_LABELS.rails],
  ['axis', PATH_VARIANT_LABELS.axis],
  ['compare', 'сравнение — все три'],
]

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
  labeling: LabelingController
  onHide: () => void
}

export function Panel({ viewer, labeling, onHide }: PanelProps) {
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
    path,
    tunnel,
    stats,
    loading,
    preset,
  } = viewer

  if (!meta) return null

  const pathReady = viewer.pathAvailable
  const tunnelReady = viewer.tunnelAvailable
  const chosen = path.source === 'compare' ? null : path.source
  const chosenMissing = pathReady && chosen !== null && !viewer.pathVariants[chosen]
  const maskLabel = viewer.maskSource ? PATH_VARIANT_LABELS[viewer.maskSource] : null
  const tunnelSummary = viewer.tunnelSummary

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
          <Hud meta={meta} idx={idx} stats={stats} tunnel={tunnelSummary} />

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

          <Group title="Линия хода">
            <CheckRow
              label="показывать"
              checked={path.visible}
              onChange={(visible) => viewer.setPath({ visible })}
              disabled={!pathReady}
            />
            <Select
              value={path.source}
              onValueChange={(next) => next && viewer.setPath({ source: String(next) as PathSource })}
              disabled={!pathReady}
            >
              <SelectTrigger className="w-full" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PATH_SOURCE_ITEMS.map(([value, label]) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <SliderRow
              label="Толщина, м"
              value={path.thickness}
              min={PATH_THICK_MIN}
              max={PATH_THICK_MAX}
              step={0.02}
              onChange={(thickness) => viewer.setPath({ thickness })}
              format={(v) => v.toFixed(2)}
              disabled={!pathReady}
            />
            {!pathReady && <Note>сервер не отдаёт path — группа неактивна</Note>}
            {chosenMissing && <Note>нет данных по «{PATH_VARIANT_LABELS[chosen as PathVariantName]}»</Note>}
          </Group>

          <Group title="Туннель безопасности">
            <CheckRow
              label="показывать"
              checked={tunnel.visible}
              onChange={(visible) => viewer.setTunnel({ visible })}
              disabled={!tunnelReady}
            />
            <SliderRow
              label="Полуширина, м"
              value={tunnel.half}
              min={TUNNEL_HALF_MIN}
              max={TUNNEL_HALF_MAX}
              step={0.01}
              onChange={(half) => viewer.setTunnel({ half })}
              format={(v) => v.toFixed(2)}
              disabled={!tunnelReady}
            />
            <SliderRow
              label="Низ, м над УГР"
              value={tunnel.low}
              min={TUNNEL_LOW_MIN}
              max={TUNNEL_LOW_MAX}
              step={0.05}
              onChange={(low) => viewer.setTunnel({ low })}
              format={(v) => v.toFixed(2)}
              disabled={!tunnelReady}
            />
            <SliderRow
              label="Верх, м над УГР"
              value={tunnel.high}
              min={TUNNEL_HIGH_MIN}
              max={TUNNEL_HIGH_MAX}
              step={0.05}
              onChange={(high) => viewer.setTunnel({ high })}
              format={(v) => v.toFixed(2)}
              disabled={!tunnelReady}
            />
            <CheckRow
              label="исключать контактный рельс"
              checked={tunnel.contact}
              onChange={(contact) => viewer.setTunnel({ contact })}
              disabled={!tunnelReady}
            />
            {tunnelSummary && (
              <Note>
                занято {tunnelSummary.binsBlocked} из {tunnelSummary.binsTotal} бинов,{' '}
                {tunnelSummary.meters.toFixed(1)} м
              </Note>
            )}
            {tunnelSummary && (
              <Note>
                в туннеле {stats.tunnelPoints} точек · маска {stats.tunnelMs.toFixed(1)} мс
              </Note>
            )}
            {tunnelReady && maskLabel && path.source === 'compare' && (
              <Note>каркас и маска — по «{maskLabel}»</Note>
            )}
            {!tunnelReady && <Note>сервер не отдаёт tunnel — группа неактивна</Note>}
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

          <Group title="Объекты">
            <ObjectsPanel
              families={viewer.objects.families}
              hidden={viewer.objects.hidden}
              error={viewer.objects.error}
              stats={viewer.objects.stats}
              onSetVisible={viewer.objects.setVisible}
              onSetAllVisible={viewer.objects.setAllVisible}
              onFit={viewer.objects.fitObjects}
              onlyObjects={viewer.objects.only}
              onSetOnlyObjects={viewer.objects.setOnly}
            />
          </Group>

          <Group title="Разметка">
            <LabelPanel labeling={labeling} frameIdx={idx} />
          </Group>

          <Group title="Легенда">
            <Legend
              meta={meta}
              pathVariants={pathReady ? viewer.pathVariants : null}
              tunnel={tunnelReady}
            />
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