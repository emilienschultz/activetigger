import { ControlsContainer, SigmaContainer, ZoomControl } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import classNames from 'classnames';
import Graph from 'graphology';
import { FC, useEffect, useMemo } from 'react';
import { FaLock } from 'react-icons/fa';
import { NodeDisplayData } from 'sigma/types';
import GraphEvents from './GraphEvents';
import { MarqueeDisplay } from './MarqueDisplay';
import { MarqueBoundingBox, MarqueeController } from './MarqueeController';

interface Props {
  data: {
    status: string;
    index: unknown[];
    x: unknown[];
    y: unknown[];
    labels: unknown[];
  };
  className?: string;
  // frameSelection
  frameSelection?: boolean;
  setFrameSelection: (frameSelection: boolean) => void;
  // bbox
  bbox?: MarqueBoundingBox;
  setBbox: (bbox?: MarqueBoundingBox) => void;
  // selection
  selectedId?: string;
  setSelectedId: (id?: string) => void;
  // color
  labelColorMapping: { [key: string]: string };
}

const sigmaStyle = { height: '500px', width: '100%' };

// Create the Component that listen to all events

export const ProjectionVizSigma: FC<Props> = ({
  data,
  className,
  frameSelection,
  setFrameSelection,
  // TODO use internal state for drawing the bbox, update frame only once dragging is over
  bbox,
  setBbox,
  selectedId,
  setSelectedId,
  labelColorMapping,
}) => {
  console.log('ProjectionVizSigma render');

  useEffect(() => {
    console.log('ProjectionVizSigma mounted');
    return () => {
      console.log('ProjectionVizSigma unmounted');
    };
  }, []);

  const graph = useMemo(() => {
    console.log('compute graph');
    const graph = new Graph<{
      x: number;
      y: number;
      labels: string;
      label: string;
      size?: number;
      color: string;
    }>();
    if (data) {
      //TODO: refine those simple heuristics
      const size = data.x.length <= 100 ? 4 : data.x.length <= 500 ? 3 : 1;
      data.x.forEach((value, index) => {
        graph.addNode(data.index[index], {
          x: value as number,
          y: data.y[index] as number,
          labels: data.labels[index] as string,
          label: data.index[index] + '',
          size,
          color: labelColorMapping[data.labels[index] as string],
        });
      });
      return graph;
    }
    return undefined;
  }, [data, labelColorMapping]);

  return (
    <div className={className}>
      <div className="d-flex align-items-center justify-content-center">
        <label
          className={classNames('d-flex align-items-center mx-4', !bbox && 'text-muted')}
          style={{ display: 'block' }}
        >
          <input
            type="checkbox"
            checked={frameSelection}
            className="mx-2"
            onChange={(e) => {
              setFrameSelection(e.target.checked);
            }}
            disabled={!bbox}
          />
          <FaLock className="me-1" /> Use current selection in annotations
          {/* Use visualisation frame to lock the selection */}
        </label>
      </div>

      <SigmaContainer
        style={sigmaStyle}
        graph={graph}
        settings={{
          allowInvalidContainer: true,
          nodeReducer: (node, data) => {
            const res: Partial<NodeDisplayData> = { ...data };

            if (selectedId === node) {
              res.highlighted = true;
            }

            return res;
          },
        }}
      >
        <GraphEvents setSelectedId={setSelectedId} />
        <ControlsContainer position={'bottom-right'}>
          <ZoomControl />
          <MarqueeController setMarqueeBoundingBox={setBbox} />
        </ControlsContainer>
        <MarqueeDisplay bbox={bbox} />
      </SigmaContainer>
    </div>
  );
};
