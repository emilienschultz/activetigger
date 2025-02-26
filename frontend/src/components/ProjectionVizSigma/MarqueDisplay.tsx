import { useRegisterEvents, useSigma } from '@react-sigma/core';
import { FC, useCallback, useEffect, useState } from 'react';
import { MarqueBoundingBox } from './MarqueeController';

export const MarqueeDisplay: FC<{ bbox?: MarqueBoundingBox }> = ({ bbox }) => {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  useEffect(() => {
    console.log('sigma changed');
  }, [sigma]);

  const [rectPosition, setRectPosition] = useState<
    { x: number; y: number; width: number; height: number } | undefined
  >(undefined);

  const updateRectPosition = useCallback(
    (bbox?: MarqueBoundingBox) => {
      if (bbox) {
        const { x, y } = sigma.graphToViewport({ x: bbox.x.min, y: bbox.y.min });
        const { x: xMax, y: yMax } = sigma.graphToViewport({ x: bbox.x.max, y: bbox.y.max });
        const width = xMax - x;
        const height = y - yMax;
        setRectPosition({ x, y: yMax, width, height });
      } else setRectPosition(undefined);
    },
    [sigma],
  );

  useEffect(() => {
    registerEvents({
      updated: () => updateRectPosition(bbox),
    });
  }, [registerEvents, updateRectPosition, bbox]);

  useEffect(() => {
    updateRectPosition(bbox);
  }, [bbox, updateRectPosition]);

  if (rectPosition === undefined) return null;
  else
    return (
      <div style={{ position: 'absolute', inset: 0 }}>
        <svg width="100%" height="100%">
          <rect
            // TODO: rotate
            x={rectPosition.x}
            y={rectPosition.y}
            width={rectPosition.width}
            height={rectPosition.height}
            stroke="black"
            fill="transparent"
            strokeWidth={2}
            strokeDasharray={6}
          />
        </svg>
      </div>
    );
};
