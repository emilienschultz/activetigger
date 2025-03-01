import { useRegisterEvents } from '@react-sigma/core';
import { Dispatch, SetStateAction, useEffect } from 'react';
import { SigmaCursorTypes } from '.';

const GraphEvents: React.FC<{
  setSelectedId: (id?: string) => void;
  setSigmaCursor: Dispatch<SetStateAction<SigmaCursorTypes>>;
}> = ({ setSelectedId, setSigmaCursor }) => {
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    // Register the events
    registerEvents({
      // (un)Select node
      clickNode: (event) => {
        setSelectedId(event.node);
      },
      clickStage: () => setSelectedId(undefined),
      // pointer cursor
      enterNode: () => {
        setSigmaCursor('pointer');
      },
      leaveNode: () => {
        setSigmaCursor(undefined);
      },
      // grabbing cursor
      mousedown: () => {
        setSigmaCursor('grabbing');
      },
      mouseup: () => {
        setSigmaCursor(undefined);
      },
    });
  }, [registerEvents, setSelectedId, setSigmaCursor]);

  return null;
};

export default GraphEvents;
