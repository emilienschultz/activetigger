import { FC, useCallback, useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { LuZoomIn } from 'react-icons/lu';
import {
  DomainTuple,
  VictoryAxis,
  VictoryChart,
  VictoryLegend,
  VictoryScatter,
  VictoryTheme,
  VictoryTooltip,
  VictoryZoomContainer,
} from 'victory';

interface Props {
  data: {
    index: number;
    x: number;
    y: number;
    labels: string[];
  }[];
  className?: string;
  // frameSelection
  frameSelection?: boolean;
  setFrameSelection: (frameSelection: boolean) => void;
  // watch zoom
  onZoom: (viewPort: { x?: DomainTuple; y?: DomainTuple }) => void;
  // selection
  selectedId?: string;
  setSelectedId: (id: string) => void;
  // color
  labelColorMapping: { [key: string]: string };
}

// zoom management
const initialZoomDomain = {
  x: [-1.5, 1.5] as DomainTuple,
  y: [-1.5, 1.5] as DomainTuple,
};
const step = 0.2;

export const UmapViz: FC<Props> = ({
  data,
  className,
  frameSelection,
  setFrameSelection,
  onZoom,
  selectedId,
  setSelectedId,
  labelColorMapping,
}) => {
  const [zoomDomain, setZoomDomain] = useState<{ x?: DomainTuple; y?: DomainTuple } | null>(
    initialZoomDomain,
  );

  const handleZoomIn = useCallback(() => {
    if (zoomDomain && zoomDomain.x && zoomDomain.y) {
      setZoomDomain({
        x: [Number(zoomDomain.x[0]) + step, Number(zoomDomain.x[1]) - step],
        y: [Number(zoomDomain.y[0]) + step, Number(zoomDomain.y[1]) - step],
      });
    }
  }, [zoomDomain]);

  const resetZoom = useCallback(() => {
    setZoomDomain(initialZoomDomain);
  }, [setZoomDomain]);

  return (
    <div className={className}>
      <div className="d-flex align-items-center justify-content-center">
        <label className="d-flex align-items-center mx-4" style={{ display: 'block' }}>
          <input
            type="checkbox"
            checked={frameSelection}
            className="mx-2"
            onChange={(_) => {
              setFrameSelection(!frameSelection);
            }}
          />
          <FaLock />
          {/* Use visualisation frame to lock the selection */}
        </label>
        <button onClick={handleZoomIn} className="btn">
          <LuZoomIn />
        </button>
        <button onClick={resetZoom}>Reset zoom</button>
      </div>
      {
        <VictoryChart
          theme={VictoryTheme.material}
          domain={initialZoomDomain}
          containerComponent={
            <VictoryZoomContainer
              zoomDomain={zoomDomain || initialZoomDomain}
              onZoomDomainChange={onZoom}
            />
          }
          height={300}
          width={300}
        >
          <VictoryAxis
            style={{
              axis: { stroke: 'transparent' },
              ticks: { stroke: 'transparent' },
              tickLabels: { fill: 'transparent' },
            }}
          />
          <VictoryScatter
            style={{
              data: {
                fill: ({ datum }) =>
                  datum.index === selectedId ? 'black' : labelColorMapping[datum.labels],
                opacity: ({ datum }) => (datum.index === selectedId ? 1 : 0.5),
                cursor: 'pointer',
                strokeWidth: 0,
              },
            }}
            size={({ datum }) => (datum.index === selectedId ? 5 : 2)}
            labels={({ datum }) => datum.index}
            labelComponent={
              <VictoryTooltip style={{ fontSize: 10 }} flyoutStyle={{ fill: 'white' }} />
            }
            data={data}
            events={[
              {
                target: 'data',
                eventHandlers: {
                  onClick: (_, props) => {
                    const { datum } = props;
                    setSelectedId(datum.index);
                  },
                },
              },
            ]}
            animate={false}
          />

          <VictoryLegend
            x={0}
            y={60}
            title="Legend"
            centerTitle
            orientation="vertical"
            gutter={10}
            style={{
              border: { stroke: 'black' },
              title: { fontSize: 5 },
              labels: { fontSize: 5 },
            }}
            data={Object.keys(labelColorMapping).map((label) => ({
              name: label,
              symbol: { fill: labelColorMapping[label] },
            }))}
          />
        </VictoryChart>
      }
    </div>
  );
};
