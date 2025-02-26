import { useRegisterEvents, useSigma } from '@react-sigma/core';
import { pick } from 'lodash';
import { FC, useCallback, useEffect, useState } from 'react';
import { Coordinates } from 'sigma/types';

import { PiSelectionBold, PiSelectionSlashBold } from 'react-icons/pi';

export interface MarqueBoundingBox {
  x: { min: number; max: number };
  y: { min: number; max: number };
}

export const MarqueeController: FC<{
  setMarqueeBoundingBox: (boundingBox?: MarqueBoundingBox) => void;
}> = ({ setMarqueeBoundingBox }) => {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  const [selectionState, setSelectionState] = useState<
    | { type: 'off' }
    | { type: 'idle' }
    | {
        type: 'marquee';
        ctrlKeyDown: boolean;
        startCorner: Coordinates;
        mouseCorner: Coordinates;
        //capturedNodes: string[];
      }
  >({ type: 'off' });

  const cleanup = useCallback(() => {
    console.log('cleanup marquee');
    sigma.getCamera().enable();
    setSelectionState({ type: 'idle' });
    setMarqueeBoundingBox(undefined);
    //TODO: setEmphasizedNodes(null);
  }, [sigma, setMarqueeBoundingBox]);

  useEffect(() => {
    const keyDownHandler = (e: KeyboardEvent) => {
      if (selectionState.type === 'idle') return;
      if (e.key === 'Escape') cleanup();
      if (e.key === 'Control') {
        setSelectionState((state) => ({ ...state, ctrlKeyDown: true }));
        // setEmphasizedNodes(
        //   new Set(selectionState.capturedNodes.concat(Array.from(selection.items))),
        // );
      }
    };
    const keyUpHandler = (e: KeyboardEvent) => {
      if (selectionState.type === 'idle') return;
      if (e.key === 'Control') {
        setSelectionState((state) => ({ ...state, ctrlKeyDown: false }));
        // setEmphasizedNodes(new Set(selectionState.capturedNodes));
      }
    };
    window.document.body.addEventListener('keydown', keyDownHandler);
    window.document.body.addEventListener('keyup', keyUpHandler);
    return () => {
      window.document.body.removeEventListener('keydown', keyDownHandler);
      window.document.body.removeEventListener('keyup', keyUpHandler);
    };
  }, [cleanup, selectionState]);

  useEffect(() => {
    registerEvents({
      mousemovebody: (e) => {
        if (selectionState.type === 'marquee') {
          const mousePosition = pick(e, 'x', 'y') as Coordinates;

          //const graph = sigma.getGraph();
          const start = sigma.viewportToGraph(selectionState.startCorner);
          const end = sigma.viewportToGraph(mousePosition);

          const minX = Math.min(start.x, end.x);
          const minY = Math.min(start.y, end.y);
          const maxX = Math.max(start.x, end.x);
          const maxY = Math.max(start.y, end.y);

          setMarqueeBoundingBox({ x: { min: minX, max: maxX }, y: { min: minY, max: maxY } });
          // TODO emphasized nodes ?
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
        if (selectionState.type !== 'off') {
          e.preventSigmaDefault();

          if (selectionState.type === 'idle') {
            console.log('start marquee');
            const mousePosition: Coordinates = pick(e.event, 'x', 'y');

            setSelectionState({
              type: 'marquee',
              startCorner: mousePosition,
              mouseCorner: mousePosition,
              ctrlKeyDown: e.event.original.ctrlKey,
              //capturedNodes: [],
            });
            sigma.getCamera().disable();
          } else {
            console.log('stop marquee');
            console.log('stop marquee');
          }
        }
      },
      click: (e) => {
        if (selectionState.type !== 'off') {
          if (selectionState.type === 'marquee') {
            e.preventSigmaDefault();
            console.log('stop marquee');
            setSelectionState({ type: 'idle' });
          }
        }
      },
    });
  }, [registerEvents, sigma, selectionState, cleanup, setMarqueeBoundingBox]);

  console.log('marquee controller render');

  return (
    <div className="react-sigma-control">
      <button
        className="btn btn-ico"
        onClick={() => {
          if (selectionState.type === 'off') {
            setSelectionState({
              type: 'idle',
            });
            sigma.getCamera().disable();
          } else {
            cleanup();
            setSelectionState({ type: 'off' });
          }
        }}
      >
        {selectionState.type !== 'off' ? <PiSelectionSlashBold /> : <PiSelectionBold />}
      </button>
    </div>
  );
};
