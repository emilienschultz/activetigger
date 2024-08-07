import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { useAppContext } from '../core/context';

export const CurrentProjectMonitoring: FC = () => {
  const { projectName } = useParams();
  const { setAppContext } = useAppContext();

  const { project, reFetch } = useProject(projectName);

  useEffect(() => {
    setAppContext((prev) => ({ ...prev, currentProject: project }));
  }, [project]);

  // Effect to poll project data regularly to monitor long lasting server tasks
  useEffect(() => {
    // execute a fetch call to update project data every 2000ms
    const intervalId = setInterval(reFetch, 2000);
    // useEffect can return a method which is executed when the component is unmounted
    return () => {
      clearInterval(intervalId);
    };
  }, [reFetch]);

  return null;
};
