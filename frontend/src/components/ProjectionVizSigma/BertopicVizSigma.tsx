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

interface NodeInputAttributesType {
  node_id: string;
  x: number;
  y: number;
  cluster_id: number;
  label: string;
}
interface NodeGraphAttributesType {
  node_id: string;
  x: number;
  y: number;
  cluster_id: number;
  label: string;
  size: number;
}

interface Props {
  nodes: NodeInputAttributesType[];
  className?: string;
  clusterIdColorMapping: { [key: string]: string };
  // selection
  setSelectedId: (id?: string) => void;
  labelColorMapping: { [key: string]: string };
  labelDescription?: { [key: string]: string };
}

const sigmaStyle = { height: '100%', width: '100%' };

export type SigmaCursorTypes = 'crosshair' | 'pointer' | 'grabbing' | undefined;
export type SigmaToolsType = 'panZoom';

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
  nodes,
  className,
  clusterIdColorMapping,
  setSelectedId,
  clusterHighlight,
  setClusterHighlightAfterDoubleClick,
}) => {
  clusterIdColorMapping['NA'] = COLORS.NA;
  // Special cursor to help interactivity affordances
  const [sigmaCursor, setSigmaCursor] = useState<SigmaCursorTypes>(undefined);

  // prepare graph for sigma from data props
  const graph = useMemo(() => {
    console.log('compute graph');
    const graph = new Graph<NodeGraphAttributesType>();
    if (nodes) {
      //TODO: refine those quick heuristics
      const size = getPointSize(nodes.length);

      nodes.forEach((node) => {
        if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
          console.log(`Skipping invalid coordinates for node ${node.node_id}`);
          return;
        }

        graph.addNode(node.node_id, { ...node, size });
      });
      return graph;
    }
    return undefined;
  }, [nodes]);

  // nodeReducer change node appearance from colorMapping and selection state
  const nodeReducer = useCallback(
    (_node: string, node: NodeGraphAttributesType): Partial<NodeDisplayData> => {
      const res: Partial<NodeDisplayData> = { ...node };
      // apply color for nodes
      res.color = labelColorMapping[data.label];

      // replace label by node id. Label is the default field in sigma to display the.. label
      if (labelDescription) {
        res.label = labelDescription[data.label];
      } else res.label = data.label;

      return res;
    },
    [clusterIdColorMapping],
  );
  const settings: Partial<Settings<NodeGraphAttributesType>> = useMemo(
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
