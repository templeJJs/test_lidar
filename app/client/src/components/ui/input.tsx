import * as React from "react"
import { cn } from "cn"

// Поле ввода панели: обычный shadcn-input, но компактнее — в панели разметки
// под поля отведено по половине строки (центр/размеры/углы идут тройками).
function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "min-w-0 rounded-md border border-input bg-transparent px-1.5 py-0.5 text-xs tabular-nums outline-none transition-[color,box-shadow] selection:bg-primary selection:text-primary-foreground file:inline-flex file:h-6 file:border-0 file:bg-transparent file:text-xs file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive dark:bg-input/30",
        className
      )}
      {...props}
    />
  )
}

export { Input }