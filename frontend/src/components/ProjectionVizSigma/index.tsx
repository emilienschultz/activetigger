import { ControlsContainer, SigmaContainer, ZoomControl } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import classNames from 'classnames';
import Graph from 'graphology';
import { FC, useCallback, useMemo, useState } from 'react';
import { PiSelectionSlashBold } from 'react-icons/pi';
import { Settings } from 'sigma/settings';
import { NodeDisplayData } from 'sigma/types';
import { Caption } from './Caption';
import GraphEvents from './GraphEvents';
import { MarqueBoundingBox, MarqueeController } from './MarqueeController';
import { MarqueeDisplay } from './MarqueeDisplay';

interface Props {
  data: {
    status: string;
    index: unknown[];
    x: unknown[];
    y: unknown[];
    labels: unknown[];
    predictions?: unknown[] | null;
  };
  className?: string;
  // bbox
  frameBbox?: MarqueBoundingBox;
  setFrameBbox: (bbox?: MarqueBoundingBox) => void;
  // selection
  selectedId?: string;
  setSelectedId: (id?: string) => void;
  // color
  labelColorMapping: { [key: string]: string };
}

const sigmaStyle = { height: '100%', width: '100%' };

export type SigmaCursorTypes = 'crosshair' | 'pointer' | 'grabbing' | undefined;
export type SigmaToolsType = 'panZoom' | 'marquee';
interface NodeAttributesType {
  x: number;
  y: number;
  label: string;
  size?: number;
}

// function to quantify point size
const getPointSize = (n: number) => {
  if (n <= 100) {
    return 8;
  } else if (n <= 500) {
    return 5;
  } else if (n <= 1000) {
    return 3;
  } else if (n <= 5000) {
    return 2;
  } else {
    return 1;
  }
};

// Create the Component that listen to all events

export const ProjectionVizSigma: FC<Props> = ({
  data,
  className,
  // get/set frame from/to app state
  frameBbox,
  setFrameBbox,
  // manage node selection
  selectedId,
  setSelectedId,
  // color dictionary
  labelColorMapping,
}) => {
  // internal bbox used by marquee. This state will be updated with setFrameBbox once drawing is done.
  // app state is used as default value
  const [bbox, setBbox] = useState<MarqueBoundingBox | undefined>(frameBbox);

  labelColorMapping['NA'] = '#ebebeb';

  // Special cursor to help interactivity affordances
  const [sigmaCursor, setSigmaCursor] = useState<SigmaCursorTypes>(undefined);
  const [activeTool, setActiveTool] = useState<SigmaToolsType>('panZoom');

  // column to use for color mapping
  const [selectedColumn, setSelectedColumn] = useState<'labels' | 'predictions'>('labels');

  // prepare graph for sigma from data props
  const graph = useMemo(() => {
    console.log('compute graph');
    const graph = new Graph<NodeAttributesType>();
    if (data) {
      //TODO: refine those simple heuristics
      const size = getPointSize(data.x.length);
      data.x.forEach((value, index) => {
        graph.addNode(data.index[index], {
          x: value as number,
          y: data.y[index] as number,
          label: data[selectedColumn]?.[index] as string,
          size,
        });
      });
      return graph;
    }
    return undefined;
  }, [data, selectedColumn]);

  // nodeReducer change node appearance from colorMapping and selection state
  const nodeReducer = useCallback(
    (node: string, data: NodeAttributesType): Partial<NodeDisplayData> => {
      const res: Partial<NodeDisplayData> = { ...data };

      if (selectedId === node) {
        // built-in appearance in Sigma which forces showing the label
        res.highlighted = true;
      }
      // apply color for nodes
      res.color = labelColorMapping[data.label];

      // replace label by node id. Label is the default field in sigma to display the.. label
      res.label = node;

      return res;
    },
    [selectedId, labelColorMapping],
  );
  const settings: Partial<Settings<NodeAttributesType>> = useMemo(
    () => ({
      allowInvalidContainer: true,
      nodeReducer,
    }),
    [nodeReducer],
  );

  return (
    <div className={className}>
      <div className="m-3">
        <label className="mx-2">Color by: </label>
        <select
          value={selectedColumn}
          onChange={(event) => {
            setSelectedColumn(event.target.value as 'labels' | 'predictions');
          }}
        >
          <option value="labels">Annotated elements</option>
          {data.predictions && <option value="predictions">Predicted elements</option>}
        </select>
      </div>
      <SigmaContainer
        className={classNames(
          sigmaCursor ? `cursor-${sigmaCursor}` : activeTool === 'marquee' && 'cursor-crosshair',
        )}
        style={sigmaStyle}
        graph={graph}
        settings={settings}
      >
        <GraphEvents setSelectedId={setSelectedId} setSigmaCursor={setSigmaCursor} />
        <ControlsContainer position="bottom-left">
          <Caption labelColorMapping={labelColorMapping} />
        </ControlsContainer>
        <ControlsContainer position={'bottom-right'}>
          <div className="border-bottom">
            {/* Active tools (zoom-pan or marquee)) buttons are managed by the marquee controller */}
            <MarqueeController
              setBbox={setBbox}
              validateBoundingBox={setFrameBbox}
              setActiveTool={setActiveTool}
            />
          </div>
          <ZoomControl />
          {/* delete bbox button */}
          {bbox !== undefined && (
            <div className="react-sigma-control">
              <button
                onClick={() => {
                  setBbox(undefined);
                  setFrameBbox(undefined);
                }}
              >
                <PiSelectionSlashBold />
              </button>
            </div>
          )}
        </ControlsContainer>
        {/* show a dashed line rectangle to render the current bbox */}
        <MarqueeDisplay bbox={bbox} />
      </SigmaContainer>
    </div>
  );
};
