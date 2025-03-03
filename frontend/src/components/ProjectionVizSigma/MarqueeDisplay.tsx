import { useRegisterEvents, useSigma } from '@react-sigma/core';
import { FC, useCallback, useEffect, useState } from 'react';
import { MarqueBoundingBox } from './MarqueeController';

export const MarqueeDisplay: FC<{ bbox?: MarqueBoundingBox }> = ({ bbox }) => {
  // sigma instance to access position transformation methods
  const sigma = useSigma();
  // sigma event hook to add update position handler
  const registerEvents = useRegisterEvents();

  // internal state: the Marquee position as SVG wants them to be
  const [rectPosition, setRectPosition] = useState<
    { x: number; y: number; width: number; height: number } | undefined
  >(undefined);

  // callback to update the SVG position according to bbox positions and sigma camera state
  const updateRectPosition = useCallback(
    (bbox?: MarqueBoundingBox) => {
      if (bbox) {
        // transform the bbox position into the viewport coordinates i.e. the coordinates we used to actually draw on the screen
        const { x, y } = sigma.graphToViewport({ x: bbox.x.min, y: bbox.y.min });
        const { x: xMax, y: yMax } = sigma.graphToViewport({ x: bbox.x.max, y: bbox.y.max });
        // SVG wants one point + width and height where bbox is two points, let's transpose
        const width = xMax - x;
        const height = y - yMax;
        if (!isNaN(x) && !isNaN(y) && !isNaN(width) && !isNaN(height))
          setRectPosition({ x, y: yMax, width, height });
        else setRectPosition(undefined);
      } else setRectPosition(undefined);
    },
    [sigma],
  );

  useEffect(() => {
    registerEvents({
      // after each sigma render we update our SVG positions to follow camera (pan & zoom)
      afterRender: () => updateRectPosition(bbox),
    });
  }, [registerEvents, updateRectPosition, bbox]);

  useEffect(() => {
    // update SVG position if bbox changes i.e. at init or when the bbox is being drawn
    updateRectPosition(bbox);
  }, [bbox, updateRectPosition]);

  // if no position, no SVG
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
