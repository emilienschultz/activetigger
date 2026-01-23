import { ControlsContainer, SigmaContainer, ZoomControl } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import classNames from 'classnames';
import Graph from 'graphology';
import { FC, useCallback, useMemo, useState } from 'react';
import { Settings } from 'sigma/settings';
import { NodeDisplayData } from 'sigma/types';
// import { Caption } from './Caption';
import { COLORS } from '../../core/colors';
import GraphEvents from './GraphEvents';

interface Props {
  data: {
    id: unknown[];
    x: unknown[];
    y: unknown[];
    cluster?: string[] | null;
  };
  className?: string;
  // selection
  setSelectedId: (id?: string) => void;
  labelColorMapping: { [key: string]: string };
  labelDescription?: { [key: string]: string };
}

const sigmaStyle = { height: '100%', width: '100%' };

export type SigmaCursorTypes = 'crosshair' | 'pointer' | 'grabbing' | undefined;
export type SigmaToolsType = 'panZoom';
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
export const BertopicVizSigma: FC<Props> = ({
  data,
  className,
  labelColorMapping,
  setSelectedId,
  labelDescription,
}) => {
  labelColorMapping['NA'] = COLORS.NA;
  // Special cursor to help interactivity affordances
  const [sigmaCursor, setSigmaCursor] = useState<SigmaCursorTypes>(undefined);

  // prepare graph for sigma from data props
  const graph = useMemo(() => {
    console.log('compute graph');
    const graph = new Graph<NodeAttributesType>();
    if (data) {
      //TODO: refine those quick heuristics
      const size = getPointSize(data.x.length);

      data.x.forEach((_, index) => {
        const x = Number(data.x[index]);
        const y = Number(data.y[index]);

        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          console.log(`Skipping invalid coordinates for node ${data.id[index]}`);
          return;
        }

        graph.addNode(data.id[index], {
          x: x,
          y: y,
          label: data.cluster?.[index] as string,
          size,
        });
      });
      return graph;
    }
    return undefined;
  }, [data]);

  // nodeReducer change node appearance from colorMapping and selection state
  const nodeReducer = useCallback(
    (_node: string, data: NodeAttributesType): Partial<NodeDisplayData> => {
      const res: Partial<NodeDisplayData> = { ...data };

      // apply color for nodes
      res.color = labelColorMapping[data.label];

      // replace label by node id. Label is the default field in sigma to display the.. label
      if (labelDescription) {
        res.label = labelDescription[data.label];
      } else res.label = data.label;

      return res;
    },
    [labelColorMapping],
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
      <SigmaContainer
        className={classNames(sigmaCursor ? `cursor-${sigmaCursor}` : '')}
        style={sigmaStyle}
        graph={graph}
        settings={settings}
      >
        <GraphEvents setSelectedId={setSelectedId} setSigmaCursor={setSigmaCursor} />
        {/* <ControlsContainer position="bottom-left">
          <Caption labelColorMapping={labelColorMapping} />
        </ControlsContainer> */}
        <ControlsContainer position={'bottom-right'}>
          <ZoomControl />
        </ControlsContainer>
      </SigmaContainer>
    </div>
  );
};
