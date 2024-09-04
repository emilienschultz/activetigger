import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { useAppContext } from '../core/context';

/**
 * Component to actualise project state in the context
 */
export const CurrentProjectMonitoring: FC = () => {
  const { projectName } = useParams();
  const { setAppContext } = useAppContext();

  const { project, reFetch } = useProject(projectName); // api call

  useEffect(() => {
    if (project && !('detail' in project)) {
      // fix BUG
      setAppContext((prev) => ({ ...prev, currentProject: project }));
    }
  }, [project, setAppContext]);

  // Effect to poll project data regularly to monitor long lasting server tasks
  // each time reFetch change
  useEffect(() => {
    //expose refetch method into context
    setAppContext((prev) => ({ ...prev, reFetchCurrentProject: reFetch }));
    // execute a fetch call to update project data every 2000ms
    const intervalId = setInterval(reFetch, 3000);
    // useEffect can return a method which is executed when the component is unmounted
    return () => {
      clearInterval(intervalId);
    };
  }, [reFetch, setAppContext]);

  return null;
};
