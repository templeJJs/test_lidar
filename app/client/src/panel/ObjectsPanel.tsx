// Панель объектов из замеров: тумблеры по семействам и происхождение каждого.
//
// Объекты приходят из `objects/specs` (маршрут /objects) — это измеренные
// элементы пути и туннеля, а не декорация. Поэтому у каждого семейства видно
// ПРОИСХОЖДЕНИЕ: «измерено» здесь, «измерено частично», «перенесено из: <запись>»
// (объект поставлен по оси, но мерялся на другой записи) или «запись не указана».
// Скрыть/показать семейство — галочкой; камера по объектам — кнопкой.

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { SwitchRow } from '@/panel/controls'
import { GroupTitle, Note } from '@/panel/controls'
import type { ObjectFamily } from '@/viewer/objects'

export interface ObjectsPanelProps {
  families: readonly ObjectFamily[]
  hidden: Readonly<Record<string, boolean>>
  error: string | null
  stats: { n_parts: number; n_vertices: number; n_faces: number } | null
  onSetVisible: (key: string, on: boolean) => void
  onSetAllVisible: (on: boolean) => void
  onFit: () => void
  onlyObjects: boolean
  onSetOnlyObjects: (on: boolean) => void
}

function provenanceBadge(family: ObjectFamily) {
  const bad = !family.accepted
  const variant = bad ? 'destructive' : family.source.kind === 'measured' ? 'secondary' : 'outline'
  const tone = bad ? '' : family.source.kind === 'gost' ? 'text-amber-400' : 'text-muted-foreground'
  return (
    <Badge variant={variant} className={`shrink-0 text-[10px] ${tone}`}>
      {family.source.text}
    </Badge>
  )
}

export function ObjectsPanel({
  families,
  hidden,
  error,
  stats,
  onSetVisible,
  onSetAllVisible,
  onFit,
  onlyObjects,
  onSetOnlyObjects,
}: ObjectsPanelProps) {
  const shown = families.filter((f) => !hidden[f.key]).length
  const allOn = families.length > 0 && shown === families.length
  const borrowed = families.filter((f) => f.source.text.startsWith('перенесено')).length
  const rejected = families.filter((f) => !f.accepted).length

  return (
    <div className="flex flex-col gap-1.5">
      <GroupTitle>Объекты из замеров</GroupTitle>
      <Note>
        Это измеренные элементы туннеля: обделка, основание пути, контактный рельс
        с оснасткой, кабельный лоток, платформа с нишами, оборудование на стенах.
        Облако точек — исходная запись лидара; объекты выведены из неё замерами.
        Чтобы рассмотреть объекты: включите «только объекты» (облако скроется,
        камера впишется по ним) и снимайте галочки по одному — так видно, что
        относится к чему. ЛКМ — поворот, ПКМ — сдвиг, колесо — зум.
      </Note>
      <SwitchRow
        label="только объекты (скрыть облако)"
        checked={onlyObjects}
        onChange={onSetOnlyObjects}
      />
      {error ? (
        <Note className="text-destructive">слой объектов: {error}</Note>
      ) : families.length === 0 ? (
        <Note>
          объектов нет: замеры ещё не сданы в objects/specs или отброшены
          (неопределённая запись, нет точек под боксом)
        </Note>
      ) : (
        <>
          <div className="flex items-center gap-2 py-0.5">
            <Checkbox
              id="objects-all"
              checked={allOn}
              onCheckedChange={(next) => onSetAllVisible(next === true)}
            />
            <Label htmlFor="objects-all" className="text-xs font-normal text-muted-foreground">
              все семейства ({shown} из {families.length})
            </Label>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="ml-auto h-6 px-2 text-[11px]"
              onClick={onFit}
            >
              вписать объекты
            </Button>
          </div>
          <Separator />
          {families.map((family) => {
            // id должен быть валидным CSS-селектором: ключ семейства --
            // «slug::часть», а часть бывает по-русски, поэтому недопустимые
            // символы заменяем на «-».
            const id = `obj-${family.key.replace(/[^\w-]+/g, '-')}`
            return (
              <div key={family.key} className="flex items-start gap-2 py-0.5">
                <Checkbox
                  id={id}
                  className="mt-0.5"
                  checked={!hidden[family.key]}
                  onCheckedChange={(next) => onSetVisible(family.key, next === true)}
                />
                <Label
                  htmlFor={id}
                  className="min-w-0 flex-1 text-xs leading-snug font-normal text-muted-foreground"
                >
                  <span className="text-foreground">{family.nameRu}</span>
                  <span className="text-muted-foreground"> · {family.part} · </span>
                  <span className="tabular-nums">{family.faces.toLocaleString('ru-RU')} тр.</span>
                  <span className="mt-0.5 block">{provenanceBadge(family)}</span>
                </Label>
              </div>
            )
          })}
          <Note>
            {stats
              ? `частей ${stats.n_parts}, треугольников ${stats.n_faces.toLocaleString('ru-RU')}`
              : 'счётчики недоступны'}
            {borrowed > 0
              ? `; перенесено из других записей: ${borrowed} — это объекты, поставленные по оси, а не замер здесь`
              : ''}
            {rejected > 0
              ? `; НЕ ПРОШЛИ ПРИЁМКУ по данным: ${rejected} — выключены, числа не подтверждены`
              : ''}
          </Note>
        </>
      )}
    </div>
  )
}