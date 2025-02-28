import { useRegisterEvents } from '@react-sigma/core';
import { useEffect } from 'react';

const GraphEvents: React.FC<{
  setSelectedId: (id?: string) => void;
}> = ({ setSelectedId }) => {
  const registerEvents = useRegisterEvents();

  /* eslint-disable no-console */
  useEffect(() => {
    // Register the events
    registerEvents({
      // node events
      clickNode: (event) => {
        setSelectedId(event.node);
      },
      // stage events
      clickStage: () => setSelectedId(undefined),
    });
  }, [registerEvents, setSelectedId]);

  return null;
};

export default GraphEvents;
