declare module 'react-heatmap-grid' {
  import { ReactNode } from 'react'

  export interface HeatmapGridProps {
    data: number[][]
    xLabels: string[]
    yLabels: string[]
    cellStyle?: (background: string, value: number, min: number, max: number, data: number[][], x: number, y: number) => React.CSSProperties
    cellHeight?: string
    cellWidth?: string
    xLabelWidth?: string
    yLabelWidth?: string
    cellRender?: (x: number, y: number, value: number) => ReactNode
    square?: boolean
    height?: number
    width?: number
  }

  const HeatmapGrid: React.FC<HeatmapGridProps>
  export default HeatmapGrid
}
