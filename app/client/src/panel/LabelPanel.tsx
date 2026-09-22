// Панель разметки: режим, ход записи, геометрия выбранного объекта, трек, список
// объектов кадра и сохранение. Логика — в useLabeling, здесь только вид.

import { useState } from 'react'
import {
  AUTOMATION_COLORS,
  AUTOMATION_LABELS,
  DEFAULT_POSE,
  LABEL_CLASSES,
  LABEL_POSES,
  type LabelClass,
  type LabelPose,
} from '@/api/label'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { GroupTitle, Note, SwitchRow } from '@/panel/controls'
import type { LabelItem, LabelingController } from '@/state/useLabeling'
import type { Vec3 } from '@/api/label'

/** Цвет статуса автоматики в hex — для кружка в списке объектов. */
function colorOf(name: LabelItem['automation']): string {
  return `#${AUTOMATION_COLORS[name].toString(16).padStart(6, '0')}`
}

function field(value: number, set: (v: number) => void, step = 0.05) {
  return (
    <Input
      type="number"
      step={step}
      value={Number.isFinite(value) ? String(Math.round(value * 1000) / 1000) : ''}
      onChange={(ev) => {
        const next = Number(ev.target.value)
        if (Number.isFinite(next)) set(next)
      }}
      className="w-full"
    />
  )
}

function Tern({
  title,
  values,
  labels,
  step,
  onChange,
}: {
  title: string
  values: Vec3
  labels: [string, string, string]
  step: number
  onChange: (v: Vec3) => void
}) {
  return (
    <div className="py-0.5">
      <div className="text-[11px] text-muted-foreground">{title}</div>
      <div className="grid grid-cols-3 gap-1">
        {labels.map((label, i) => (
          <div key={label} className="min-w-0">
            <div className="text-[10px] text-muted-foreground">{label}</div>
            {field(values[i], (v) => {
              const next: Vec3 = [values[0], values[1], values[2]]
              next[i] = v
              onChange(next)
            }, step)}
          </div>
        ))}
      </div>
    </div>
  )
}

export function LabelPanel({ labeling, frameIdx }: { labeling: LabelingController; frameIdx: number }) {
  const [framesText, setFramesText] = useState('')
  const { selected } = labeling
  const motion = labeling.motion

  return (
    <div className="flex flex-col gap-1.5">
      <GroupTitle>Разметка</GroupTitle>
      <SwitchRow
        label="режим разметки (клик по облаку — предложение)"
        checked={labeling.on}
        onChange={labeling.setOn}
      />
      {!labeling.on ? (
        <Note>
          Режим не мешает просмотру: пока он выключен, клик и мышь работают как
          обычно. Включите — и ЛКМ по облаку попросит у сервера габарит объекта.
        </Note>
      ) : (
        <>
          <Note>
            ЛКМ по объекту — предложение габарита. Тумблер «правка мышью» ниже
            замораживает камеру: тогда ЛКМ тянет выбранный бокс по полу. Точки
            объекта считает сервер ТРАССИРОВКОЙ лучей кадра — то, что видно
            голубым, и есть то, что запишется в <code>.npz</code>.
          </Note>

          <Note className={motion?.measured === false ? 'text-amber-400' : undefined}>
            {labeling.motionError
              ? `ход записи: ${labeling.motionError}`
              : motion
                ? `ход записи (замер по стене): ${motion.m_per_frame.toFixed(3)} м/кадр · ` +
                  `${motion.kmh.toFixed(1)} км/ч · остаток ${motion.residual_m?.toFixed(3) ?? '—'} м · ` +
                  `кадры ${motion.pairs.map((p) => p.frames.join('→')).join(', ')}`
                : 'ход записи: считается…'}
          </Note>

          <SwitchRow
            label="правка мышью (тянуть бокс по полу)"
            checked={labeling.dragMode}
            onChange={labeling.setDragMode}
            disabled={!selected}
          />

          <Separator />

          {!selected ? (
            <Note>
              объектов нет: кликните по облаку в режиме разметки. Если автоматика
              не предложит габарит, объект встанет вручную (красный) — правьте
              числами.
            </Note>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span
                  className="size-2.5 shrink-0 rounded-full"
                  style={{ background: colorOf(selected.automation) }}
                />
                <Badge variant={selected.automation === 'detector' ? 'secondary' : 'outline'} className="text-[10px]">
                  {AUTOMATION_LABELS[selected.automation]}
                </Badge>
                <span className="ml-auto text-[10px] text-muted-foreground tabular-nums">
                  {selected.id ?? 'не сохранён'}
                </span>
              </div>
              <Note>{selected.reason}</Note>

              <div className="grid grid-cols-2 gap-2">
                <div className="py-0.5">
                  <div className="text-[11px] text-muted-foreground">класс</div>
                  <Select
                    value={selected.cls}
                    onValueChange={(v) =>
                      labeling.patch(selected.key, {
                        cls: v as LabelClass,
                        pose: DEFAULT_POSE[v as LabelClass],
                      })
                    }
                  >
                    <SelectTrigger size="sm" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LABEL_CLASSES.map(([key, label]) => (
                        <SelectItem key={key} value={key}>
                          {label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="py-0.5">
                  <div className="text-[11px] text-muted-foreground">поза</div>
                  <Select
                    value={selected.pose}
                    onValueChange={(v) => labeling.patch(selected.key, { pose: v as LabelPose })}
                    disabled={selected.cls !== 'person'}
                  >
                    <SelectTrigger size="sm" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LABEL_POSES.map(([key, label]) => (
                        <SelectItem key={key} value={key}>
                          {label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Tern
                title="центр, м (X поперёк, Y вдоль, Z вверх)"
                values={selected.center}
                labels={['X', 'Y', 'Z']}
                step={0.05}
                onChange={(v) => labeling.patch(selected.key, { center: v })}
              />
              <Tern
                title="размеры, м (ширина, высота, длина)"
                values={selected.size}
                labels={['шир', 'выс', 'дл']}
                step={0.05}
                onChange={(v) => labeling.patch(selected.key, { size: v })}
              />
              <Tern
                title="углы, ° (рыскание, тангаж, крен)"
                values={[selected.yaw, selected.pitch, selected.roll]}
                labels={['рыск', 'танг', 'крен']}
                step={1}
                onChange={(v) =>
                  labeling.patch(selected.key, { yaw: v[0], pitch: v[1], roll: v[2] })
                }
              />

              <div className="grid grid-cols-2 gap-2">
                <div className="py-0.5">
                  <div className="text-[11px] text-muted-foreground">опорный кадр</div>
                  {field(selected.refFrame, (v) =>
                    labeling.patch(selected.key, {
                      refFrame: Math.round(v),
                      frames: selected.frames.includes(Math.round(v))
                        ? selected.frames
                        : [Math.round(v)],
                    }), 1)}
                </div>
                <div className="py-0.5">
                  <div className="text-[11px] text-muted-foreground">кадры (через запятую)</div>
                  <Input
                    type="text"
                    data-field="frames"
                    value={framesText === '' ? selected.frames.join(', ') : framesText}
                    onChange={(ev) => {
                      setFramesText(ev.target.value)
                      const list = ev.target.value
                        .split(',')
                        .map((v) => v.trim())
                        .filter((v) => v !== '' && Number.isFinite(Number(v)))
                        .map((v) => Math.round(Number(v)))
                      if (list.length) labeling.patch(selected.key, { frames: list })
                    }}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="py-0.5">
                <div className="text-[11px] text-muted-foreground">
                  скорость трека, км/ч (замерено {labeling.measurementKmh.toFixed(1)})
                </div>
                <div className="flex items-center gap-1">
                  {field(labeling.speedKmhOf(selected), (v) =>
                    labeling.patch(selected.key, { speedKmh: v, speedSource: 'manual' }), 1)}
                  <Button
                    size="xs"
                    variant={selected.speedSource === 'measured' ? 'secondary' : 'outline'}
                    onClick={() =>
                      labeling.patch(selected.key, { speedKmh: null, speedSource: 'measured' })
                    }
                    title="вернуть замеренный ход записи"
                  >
                    замер
                  </Button>
                  <Button
                    size="xs"
                    variant="outline"
                    onClick={() => labeling.patch(selected.key, { speedKmh: 50, speedSource: 'manual' })}
                    title="быстрый подход: 50 км/ч"
                  >
                    50
                  </Button>
                </div>
                <Note>
                  объект стоит на пути, а сенсор едет: y(f) = y_ref + v·(f − кадр_ref).
                  {(labeling.speedKmhOf(selected) / 3.6 / (labeling.motion?.hz || 10)).toFixed(4)} м/кадр.
                </Note>
              </div>

              <Separator />

              <GroupTitle>Точки трассировки</GroupTitle>
              {labeling.previewError ? (
                <Note className="text-destructive">{labeling.previewError}</Note>
              ) : labeling.previewBusy && !labeling.preview ? (
                <Note>считаю лучи кадра…</Note>
              ) : labeling.preview ? (
                <Note>
                  точек {labeling.preview.n_points} (лучей попало{' '}
                  {labeling.preview.counts.rays_hit} из {labeling.preview.counts.rays}, заслонено{' '}
                  {labeling.preview.counts.points_removed}) · дальность{' '}
                  {labeling.preview.path?.along_m.toFixed(2) ?? '—'} м, поперёк{' '}
                  {labeling.preview.path?.lateral_m.toFixed(2) ?? '—'} м ·{' '}
                  <span
                    className={
                      labeling.preview.confirmed ? 'text-green-400' : 'text-amber-400'
                    }
                  >
                    {labeling.preview.confirmed
                      ? 'детектор видит объект на дополненном облаке'
                      : 'детектор объект не подтвердил'}
                  </span>
                </Note>
              ) : (
                <Note>предпросмотр не запрошен</Note>
              )}
              <Button size="xs" variant="outline" onClick={labeling.refreshPreview}>
                пересчитать точки
              </Button>
            </>
          )}

          <Separator />

          <GroupTitle>Объекты кадра ({labeling.frameObjects.length})</GroupTitle>
          {labeling.objects.length === 0 ? (
            <Note>пока пусто</Note>
          ) : (
            <div className="flex flex-col gap-0.5">
              {labeling.objects.map((o) => {
                const here = o.frames.includes(frameIdx)
                return (
                  <div key={o.key} className="flex items-center gap-1.5 py-0.5">
                    <span
                      className="size-2.5 shrink-0 rounded-full"
                      style={{ background: colorOf(o.automation) }}
                      title={AUTOMATION_LABELS[o.automation]}
                    />
                    <button
                      type="button"
                      onClick={() => labeling.select(o.key)}
                      className={`min-w-0 flex-1 truncate text-left text-[11px] ${
                        o.key === labeling.selectedKey ? 'text-foreground' : 'text-muted-foreground'
                      }`}
                      title="выбрать объект"
                    >
                      {o.cls} · вдоль {(-o.center[1]).toFixed(1)} м · кадр {o.refFrame}
                      {o.frames.length > 1 ? `+${o.frames.length - 1}` : ''} ·{' '}
                      {here ? '' : 'не в кадре'}
                    </button>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => labeling.remove(o.key)}
                      title="удалить объект"
                    >
                      ×
                    </Button>
                  </div>
                )
              })}
            </div>
          )}
          <LabelLegend />

          <div className="flex items-center gap-1 pt-1">
            <Button
              size="sm"
              onClick={() => void labeling.save()}
              disabled={labeling.saveBusy || labeling.objects.length === 0}
            >
              {labeling.saveBusy ? 'сохраняю…' : 'Сохранить'}
            </Button>
            <Note>→ labels/index.jsonl, карточки и .npz</Note>
          </div>

          {labeling.saveReport && (
            <Note className={labeling.saveReport.errors.length ? 'text-destructive' : undefined}>
              сохранено {labeling.saveReport.written.length} объектов:
              {' '}всего в записи {labeling.saveReport.objects_total}
              {labeling.saveReport.written[0]
                ? `, точек ${labeling.saveReport.written[0].n_points}, npz ${labeling.saveReport.written[0].size_bytes} Б`
                : ''}
              {labeling.saveReport.errors.length
                ? `; ошибки: ${labeling.saveReport.errors.map((e) => e.error).join('; ')}`
                : ''}
            </Note>
          )}

          {labeling.status && <Note className="text-amber-400">{labeling.status}</Note>}
        </>
      )}
    </div>
  )
}

/** Строка легенды цветов статуса — показывается под списком объектов. */
export function LabelLegend() {
  return (
    <div className="flex flex-wrap gap-x-2 gap-y-0.5 pt-1">
      {(['detector', 'cluster', 'manual'] as const).map((key) => (
        <span key={key} className="flex items-center gap-1 text-[10px] text-muted-foreground">
          <span className="size-2 rounded-full" style={{ background: colorOf(key) }} />
          {AUTOMATION_LABELS[key]}
        </span>
      ))}
    </div>
  )
}