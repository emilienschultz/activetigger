import { useRegisterEvents, useSigma } from '@react-sigma/core';
import { pick } from 'lodash';
import { Dispatch, FC, SetStateAction, useCallback, useEffect, useState } from 'react';
import { Coordinates } from 'sigma/types';

import classNames from 'classnames';
import { PiCursorFill, PiSelectionBold } from 'react-icons/pi';
import { SigmaToolsType } from '.';

export interface MarqueBoundingBox {
  x: { min: number; max: number };
  y: { min: number; max: number };
}

export const MarqueeController: FC<{
  setBbox: Dispatch<SetStateAction<MarqueBoundingBox | undefined>>;
  validateBoundingBox: (boundingBox?: MarqueBoundingBox) => void;
  setActiveTool: Dispatch<SetStateAction<SigmaToolsType>>;
}> = ({ setBbox, validateBoundingBox, setActiveTool }) => {
  // sigma hooks
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();

  // internal state
  const [selectionState, setSelectionState] = useState<
    | { type: 'off' }
    | { type: 'idle' }
    | {
        type: 'marquee';
        startCorner: Coordinates;
        mouseCorner: Coordinates;
        //capturedNodes: string[];
      }
  >({ type: 'off' });

  // cleaning state when marquee closes
  const backToIdle = useCallback(() => {
    console.log('closing marquee');
    sigma.getCamera().enable();
    setSelectionState({ type: 'idle' });
    //TODO: setEmphasizedNodes(null);
  }, [sigma]);

  const closeMarkee = useCallback(() => {
    console.log('stop marquee');
    setBbox((prev) => {
      setTimeout(() => validateBoundingBox(prev), 0);
      return prev;
    });
    backToIdle();
  }, [validateBoundingBox, setBbox, backToIdle]);

  // Keyboard events
  useEffect(() => {
    const keyDownHandler = (e: KeyboardEvent) => {
      if (selectionState.type === 'idle') return;
      if (selectionState.type === 'marquee' && e.key === 'Escape') {
        setBbox(undefined);
        validateBoundingBox(undefined);
        backToIdle();
      }
    };

    window.document.body.addEventListener('keydown', keyDownHandler);
    return () => {
      window.document.body.removeEventListener('keydown', keyDownHandler);
    };
  }, [backToIdle, selectionState, validateBoundingBox, setBbox]);

  useEffect(() => {
    registerEvents({
      mousemovebody: (e) => {
        // update bbox if ongoing marquee drawing
        if (selectionState.type === 'marquee') {
          const mousePosition = pick(e, 'x', 'y') as Coordinates;

          //const graph = sigma.getGraph();
          const start = sigma.viewportToGraph(selectionState.startCorner);
          const end = sigma.viewportToGraph(mousePosition);

          const minX = Math.min(start.x, end.x);
          const minY = Math.min(start.y, end.y);
          const maxX = Math.max(start.x, end.x);
          const maxY = Math.max(start.y, end.y);

          // update bbox state to update marquee display
          setBbox({ x: { min: minX, max: maxX }, y: { min: minY, max: maxY } });

          // TODO emphasized nodes ? If we want to highlight nodes in the bbox
          // const capturedNodes = graph.filterNodes((node, { x, y }) => {
          //   const size = sigma.getNodeDisplayData(node)!.size as number;
          //   return !(x + size < minX || x - size > maxX || y + size < minY || y - size > maxY);
          // });
          // setEmphasizedNodes(
          //   new Set(
          //     capturedNodes.concat(
          //       selectionState.ctrlKeyDown && selection.type === 'nodes'
          //         ? Array.from(selection.items)
          //         : [],
          //     ),
          //   ),
          // );

          setSelectionState({
            ...selectionState,
            mouseCorner: mousePosition,
          });
        }
      },
      clickStage: (e) => {
        // start / stop Marquee drawing
        if (selectionState.type !== 'off') {
          e.preventSigmaDefault();

          if (selectionState.type === 'idle') {
            console.log('start marquee');
            const mousePosition: Coordinates = pick(e.event, 'x', 'y');

            setSelectionState({
              type: 'marquee',
              startCorner: mousePosition,
              mouseCorner: mousePosition,
              //capturedNodes: [],
            });
            sigma.getCamera().disable();
          } else {
            closeMarkee();
          }
        }
      },
      click: (e) => {
        // to make sur a click elsewhere than stage closes the marquee
        if (selectionState.type === 'marquee') {
          e.preventSigmaDefault();
          closeMarkee();
        }
      },
    });
  }, [registerEvents, sigma, selectionState, backToIdle, setBbox, closeMarkee]);

  return (
    <>
      <div className="react-sigma-control">
        {/* normal zoom-pan tool activation button*/}
        <button
          className={classNames(
            selectionState.type === 'off' ? 'bg-primary text-light' : 'cursor-pointer',
          )}
          disabled={selectionState.type === 'off'}
          onClick={() => {
            if (selectionState.type !== 'off') {
              setSelectionState({ type: 'off' });
              setActiveTool('panZoom');
            }
          }}
        >
          <PiCursorFill />
        </button>
      </div>
      <div className="react-sigma-control">
        {/* normal marquee tool activation button */}
        <button
          className={classNames(
            selectionState.type !== 'off' ? 'bg-primary text-light' : 'cursor-pointer',
          )}
          disabled={selectionState.type !== 'off'}
          onClick={() => {
            if (selectionState.type === 'off') {
              setSelectionState({
                type: 'idle',
              });
              setActiveTool('marquee');
            }
          }}
        >
          <PiSelectionBold />
        </button>{' '}
      </div>
    </>
  );
};
